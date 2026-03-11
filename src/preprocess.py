from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import Counter
import pandas as pd
import random
import copy
import math

# Теги в train.csv заданы, как числа, так что преобразуем их по коду
# из раздела Data в соревновании на кегле
ID2TAG: Dict[int, str] = {
    0: "O",
    1: "B-per",
    2: "I-per",
    3: "B-gpe",
    4: "I-gpe",
    5: "B-eve",
    6: "I-eve",
    7: "B-geo",
    8: "I-geo",
    9: "B-nat",
    10: "I-nat",
    11: "B-art",
    12: "I-art",
    13: "B-tim",
    14: "I-tim",
    15: "B-org",
    16: "I-org",
}
TAG2ID: Dict[str, int] = {v: k for k, v in ID2TAG.items()}


@dataclass
class SentenceExample:
    sentence_id: int
    row_ids: List[int]
    tokens: List[str]
    pos_tags: List[str]
    ner_tags: Optional[List[str]] = None


# Аугментируем тупым образом данные (через копирование)
def statistical_oversample(train_examples: List[SentenceExample], max_duplicates: int = 15) -> List[SentenceExample]:
    entity_counts = Counter()
    for ex in train_examples:
        if ex.ner_tags:
            for tag in ex.ner_tags:
                if tag.startswith("B-"):
                    entity_counts[tag] += 1

    if not entity_counts:
        return train_examples

    max_count = max(entity_counts.values())

    class_weights = {}
    for tag, count in entity_counts.items():
        class_weights[tag] = min(max_count / count, max_duplicates)

    augmented_data = []
    random.seed(42)

    for ex in train_examples:
        if not ex.ner_tags or all(t == "O" for t in ex.ner_tags):
            sentence_weight = 1.0
        else:
            weights_in_sentence = [class_weights[t] for t in ex.ner_tags if t.startswith("B-")]
            sentence_weight = max(weights_in_sentence) if weights_in_sentence else 1.0

        n_copies = int(math.floor(sentence_weight))
        for _ in range(n_copies):
            augmented_data.append(copy.deepcopy(ex))

    random.shuffle(augmented_data)
    return augmented_data


def load_csv(path: str, has_labels: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)

    # если был кривой импорт, фиксим
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.rename(columns={df.columns[0]: "row_id"})
    else:
        df = df.reset_index(names="row_id")

    # для каждого следующего слова тянем Sentence_id вниз,
    # пока не появится новое значение
    # т.к. в датасете у нас пустые места проставлены для всех
    # слов, кроме первого
    df["Sentence_id"] = df["Sentence_id"].ffill()
    df["Sentence_id"] = df["Sentence_id"].astype(int)

    # для пропавших слов просто заполняем неизвестным
    df["Word"] = df["Word"].fillna("").astype(str)
    df["POS"] = df["POS"].fillna("UNK").astype(str)

    if has_labels:
        df["Tag"] = df["Tag"].astype(int)
        df["Tag_str"] = df["Tag"].map(ID2TAG)

    return df


def build_sentence_examples(df: pd.DataFrame, has_labels: bool = True) -> List[SentenceExample]:
    examples: List[SentenceExample] = []

    # собираем предложения отдельно в примеры для обучения
    # группируем слова по предложениям, потом просто распихиваем в датакласс
    for sid, group in df.groupby("Sentence_id", sort=True):
        row_ids = group["row_id"].tolist()
        tokens = group["Word"].tolist()
        pos_tags = group["POS"].tolist()

        ner_tags = None
        if has_labels:
            ner_tags = group["Tag_str"].tolist()

        examples.append(
            SentenceExample(
                sentence_id=int(sid),
                row_ids=row_ids,
                tokens=tokens,
                pos_tags=pos_tags,
                ner_tags=ner_tags,
            )
        )

    return examples


def train_valid_split_balanced(
    examples,
    valid_size: float = 0.2,
    random_state: int = 42,
):
    rng = random.Random(random_state)

    # собираем примеры, которые содержат хотя бы одну сущность
    with_entities = [x for x in examples if any(tag != "O" for tag in (x.ner_tags or []))]
    # примеры без сущностей
    without_entities = [x for x in examples if all(tag == "O" for tag in (x.ner_tags or []))]

    rng.shuffle(with_entities)
    rng.shuffle(without_entities)

    split_with = int(len(with_entities) * (1.0 - valid_size))
    split_without = int(len(without_entities) * (1.0 - valid_size))

    # Берём равномерно от каждого типа примеров по 80/20 данных
    train_examples = with_entities[:split_with] + without_entities[:split_without]
    valid_examples = with_entities[split_with:] + without_entities[split_without:]

    rng.shuffle(train_examples)
    rng.shuffle(valid_examples)

    return train_examples, valid_examples

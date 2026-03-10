

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


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


def fix_iob2(tags: List[str]) -> List[str]:
    """
    Repair invalid IOB2 sequences:
    O -> I-x becomes B-x
    B-a -> I-b becomes B-b
    """
    fixed: List[str] = []
    prev_tag = "O"

    for tag in tags:
        if tag == "O":
            fixed.append(tag)
            prev_tag = tag
            continue

        prefix, ent = tag.split("-", 1)

        if prefix == "B":
            fixed.append(tag)
            prev_tag = tag
            continue

        # prefix == "I"
        if prev_tag == "O":
            fixed.append(f"B-{ent}")
        else:
            prev_prefix, prev_ent = prev_tag.split("-", 1)
            if prev_ent != ent:
                fixed.append(f"B-{ent}")
            else:
                fixed.append(tag)

        prev_tag = fixed[-1]

    return fixed


def load_csv(path: str, has_labels: bool = True) -> pd.DataFrame:
    
    df = pd.read_csv(path)

    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.rename(columns={df.columns[0]: "row_id"})
    else:
        df = df.reset_index(names="row_id")

    df["Sentence_id"] = df["Sentence_id"].ffill()
    df["Sentence_id"] = df["Sentence_id"].astype(int)

    df["Word"] = df["Word"].fillna("").astype(str)
    df["POS"] = df["POS"].fillna("UNK").astype(str)

    if has_labels:
        df["Tag"] = df["Tag"].astype(int)
        df["Tag_str"] = df["Tag"].map(ID2TAG)

    return df


def build_sentence_examples(df: pd.DataFrame, has_labels: bool = True) -> List[SentenceExample]:
    examples: List[SentenceExample] = []

    for sid, group in df.groupby("Sentence_id", sort=True):
        row_ids = group["row_id"].tolist()
        tokens = group["Word"].tolist()
        pos_tags = group["POS"].tolist()

        if has_labels:
            ner_tags = group["Tag_str"].tolist()
            ner_tags = fix_iob2(ner_tags)
        else:
            ner_tags = None

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

# def train_valid_split(
#     examples: List[SentenceExample],
#     valid_size: float = 0.2,
#     random_state: int = 42,
# ) -> tuple[List[SentenceExample], List[SentenceExample]]:
#     import random

#     rng = random.Random(random_state)
#     items = examples[:]
#     rng.shuffle(items)

#     split_idx = int(len(items) * (1.0 - valid_size))
#     train_examples = items[:split_idx]
#     valid_examples = items[split_idx:]

#     return train_examples, valid_examples

def train_valid_split_balanced(
    examples,
    valid_size: float = 0.2,
    random_state: int = 42,
):
    import random

    rng = random.Random(random_state)

    with_entities = [x for x in examples if any(tag != "O" for tag in (x.ner_tags or []))]
    without_entities = [x for x in examples if all(tag == "O" for tag in (x.ner_tags or []))]

    rng.shuffle(with_entities)
    rng.shuffle(without_entities)

    split_with = int(len(with_entities) * (1.0 - valid_size))
    split_without = int(len(without_entities) * (1.0 - valid_size))

    train_examples = with_entities[:split_with] + without_entities[:split_without]
    valid_examples = with_entities[split_with:] + without_entities[split_without:]

    rng.shuffle(train_examples)
    rng.shuffle(valid_examples)

    return train_examples, valid_examples
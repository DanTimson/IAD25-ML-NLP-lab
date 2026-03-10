from __future__ import annotations

from src.crf import CRFNER
from src.preprocess import build_sentence_examples, load_csv, train_valid_split_balanced
from src.metrics import entity_level_prf1


def main() -> None:
    df = load_csv("data/train.csv", has_labels=True)
    examples = build_sentence_examples(df, has_labels=True)

    train_examples, valid_examples = train_valid_split_balanced(
        examples,
        valid_size=0.2,
        random_state=42,
    )

    print(f"Total sentences: {len(examples)}")
    print(f"Train sentences: {len(train_examples)}")
    print(f"Valid sentences: {len(valid_examples)}")

    model = CRFNER()
    model.fit(train_examples)

    y_true = [x.ner_tags for x in valid_examples]
    y_pred = model.predict(valid_examples)

    scores = entity_level_prf1(y_true, y_pred)
    print("Validation scores:")
    for k, v in scores.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
from __future__ import annotations

from src.preprocess import build_sentence_examples, load_csv, train_valid_split_balanced
from src.tf import TransformerConfig, TransformerNER


def main() -> None:
    df = load_csv("data/train.csv", has_labels=True)
    examples = build_sentence_examples(df, has_labels=True)

    train_examples, valid_examples = train_valid_split_balanced(
        examples,
        valid_size=0.2,
        random_state=42,
    )

    config = TransformerConfig(
        model_name="distilbert-base-cased",
        max_length=128,
        output_dir="outputs/distilbert_ner",
        learning_rate=2e-5,
        train_batch_size=16,
        eval_batch_size=16,
        num_train_epochs=3,
    )

    model = TransformerNER(config)
    model.fit(train_examples, valid_examples)


if __name__ == "__main__":
    main()
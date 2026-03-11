import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from src.preprocess import ID2TAG, TAG2ID, build_sentence_examples, load_csv


def predict_test_tags(
    model_path: str,
    test_examples,
    max_length: int = 128,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    all_predictions = []

    for ex in test_examples:
        enc = tokenizer(
            ex.tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits[0]
        pred_ids = logits.argmax(dim=-1).cpu().tolist()

        word_ids = tokenizer(
            ex.tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        ).word_ids()

        seq_pred = []
        prev_word_idx = None
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != prev_word_idx:
                seq_pred.append(pred_ids[token_idx])
            prev_word_idx = word_idx

        if len(seq_pred) != len(ex.tokens):
            seq_pred = seq_pred[: len(ex.tokens)]

        all_predictions.append(seq_pred)

    return all_predictions


def main() -> None:
    df_test = load_csv("data/test.csv", has_labels=False)
    test_examples = build_sentence_examples(df_test, has_labels=False)

    pred_seqs = predict_test_tags(
        model_path="outputs/distilbert_ner/checkpoint-3838",
        test_examples=test_examples,
        max_length=128,
    )

    rows = []
    for ex, pred_ids in zip(test_examples, pred_seqs):
        for row_id, tag_id in zip(ex.row_ids, pred_ids):
            rows.append({"ID": row_id, "TARGET": int(tag_id)})

    submission = pd.DataFrame(rows).sort_values("ID")
    submission.to_csv("submission.csv", index=False)
    print(submission.head())
    print("Saved submission.csv")


if __name__ == "__main__":
    main()



from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from src.preprocess import ID2TAG, TAG2ID, SentenceExample
from src.metrics import entity_level_prf1
from tqdm.auto import tqdm


@dataclass
class TransformerConfig:
    model_name: str = "distilbert-base-cased"
    max_length: int = 128
    output_dir: str = "outputs/distilbert_ner"
    learning_rate: float = 2e-5
    train_batch_size: int = 16
    eval_batch_size: int = 16
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    show_progress: bool = True


def examples_to_hf_dataset(examples: List[SentenceExample]) -> Dataset:
    data = {
        "tokens": [x.tokens for x in examples],
        "pos_tags": [x.pos_tags for x in examples],
        "ner_tags": [[TAG2ID[t] for t in x.ner_tags] for x in examples],
    }
    return Dataset.from_dict(data)


class TransformerNER:
    def __init__(self, config: TransformerConfig) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            config.model_name,
            num_labels=len(ID2TAG),
            id2label=ID2TAG,
            label2id=TAG2ID,
        )
        self.trainer: Optional[Trainer] = None

    def tokenize_and_align_labels(self, examples: Dict[str, List[List[str]]]) -> Dict[str, List[List[int]]]:
        tokenized = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=self.config.max_length,
        )

        labels = []

        for i, label_seq in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_seq[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized["labels"] = labels
        return tokenized

    def _build_trainer(self, train_tok=None, valid_tok=None) -> Trainer:
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        args = TrainingArguments(
            output_dir=self.config.output_dir,
            eval_strategy="epoch" if valid_tok is not None else "no",
            save_strategy="epoch" if valid_tok is not None else "no",
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            num_train_epochs=self.config.num_train_epochs,
            weight_decay=self.config.weight_decay,
            load_best_model_at_end=valid_tok is not None,
            metric_for_best_model="f1" if valid_tok is not None else None,
            greater_is_better=True if valid_tok is not None else None,
            save_total_limit=2,
            report_to="none",
            disable_tqdm=not self.config.show_progress,
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)

            true_tags = []
            pred_tags = []

            for pred_seq, label_seq in zip(predictions, labels):
                cur_true = []
                cur_pred = []

                for p, l in zip(pred_seq, label_seq):
                    if l == -100:
                        continue
                    cur_true.append(ID2TAG[int(l)])
                    cur_pred.append(ID2TAG[int(p)])

                true_tags.append(cur_true)
                pred_tags.append(cur_pred)

            scores = entity_level_prf1(true_tags, pred_tags)
            return {
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["f1"],
            }

        trainer_kwargs = dict(
            model=self.model,
            args=args,
            data_collator=data_collator,
            compute_metrics=compute_metrics if valid_tok is not None else None,
        )

        if train_tok is not None:
            trainer_kwargs["train_dataset"] = train_tok
        if valid_tok is not None:
            trainer_kwargs["eval_dataset"] = valid_tok

        try:
            trainer_kwargs["processing_class"] = self.tokenizer
            trainer = Trainer(**trainer_kwargs)
        except TypeError:
            trainer_kwargs.pop("processing_class", None)
            trainer = Trainer(**trainer_kwargs)

        self.trainer = trainer
        return trainer

    def fit(
        self,
        train_examples: List[SentenceExample],
        valid_examples: Optional[List[SentenceExample]] = None,
        do_train: bool = True,
    ) -> Trainer:
        train_ds = examples_to_hf_dataset(train_examples)
        train_tok = train_ds.map(self.tokenize_and_align_labels, batched=True)

        valid_tok = None
        if valid_examples is not None:
            valid_ds = examples_to_hf_dataset(valid_examples)
            valid_tok = valid_ds.map(self.tokenize_and_align_labels, batched=True)

        trainer = self._build_trainer(train_tok=train_tok, valid_tok=valid_tok)

        if do_train:
            trainer.train()
            self.save(self.config.output_dir)

        return trainer

    def save(self, save_dir: Optional[str] = None) -> None:
        save_dir = save_dir or self.config.output_dir
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    def load(self, checkpoint_dir: str) -> None:
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        self._build_trainer(train_tok=None, valid_tok=None)

    def predict(self, examples):
        import torch

        if self.trainer is None:
            self._build_trainer(train_tok=None, valid_tok=None)

        self.model.eval()
        device = next(self.model.parameters()).device

        all_preds = []

        for ex in tqdm(
            examples,
            desc="Transformer predict",
            disable=not self.config.show_progress,
        ):
            enc = self.tokenizer(
                ex.tokens,
                truncation=True,
                is_split_into_words=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                outputs = self.model(**enc)

            pred_ids = outputs.logits.argmax(dim=-1)[0].detach().cpu().tolist()

            enc_no_tensors = self.tokenizer(
                ex.tokens,
                truncation=True,
                is_split_into_words=True,
                max_length=self.config.max_length,
            )
            word_ids = enc_no_tensors.word_ids()

            seq_preds = ["O"] * len(ex.tokens)

            seen = set()
            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                if word_idx not in seen:
                    seq_preds[word_idx] = ID2TAG[int(pred_ids[token_idx])]
                    seen.add(word_idx)

            all_preds.append(seq_preds)

        return all_preds
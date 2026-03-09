from __future__ import annotations

from typing import Dict, List

import joblib
import sklearn_crfsuite

from src.preprocess import SentenceExample
from tqdm.auto import tqdm


def word2features(tokens: List[str], pos_tags: List[str], i: int) -> Dict[str, object]:
    word = tokens[i]
    pos = pos_tags[i]

    features: Dict[str, object] = {
        "bias": 1.0,
        "word.lower": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word[:3]": word[:3],
        "word[:2]": word[:2],
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.hasdigit": any(ch.isdigit() for ch in word),
        "word.hashyphen": "-" in word,
        "len(word)": len(word),
        "pos": pos,
        "pos[:2]": pos[:2],
    }

    if i > 0:
        prev_word = tokens[i - 1]
        prev_pos = pos_tags[i - 1]
        features.update({
            "-1:word.lower": prev_word.lower(),
            "-1:word.istitle": prev_word.istitle(),
            "-1:word.isupper": prev_word.isupper(),
            "-1:pos": prev_pos,
            "-1:pos[:2]": prev_pos[:2],
        })
    else:
        features["BOS"] = True

    if i < len(tokens) - 1:
        next_word = tokens[i + 1]
        next_pos = pos_tags[i + 1]
        features.update({
            "+1:word.lower": next_word.lower(),
            "+1:word.istitle": next_word.istitle(),
            "+1:word.isupper": next_word.isupper(),
            "+1:pos": next_pos,
            "+1:pos[:2]": next_pos[:2],
        })
    else:
        features["EOS"] = True

    return features


def sent2features(example: SentenceExample) -> List[Dict[str, object]]:
    return [word2features(example.tokens, example.pos_tags, i) for i in range(len(example.tokens))]


def sent2labels(example: SentenceExample) -> List[str]:
    if example.ner_tags is None:
        raise ValueError("SentenceExample has no labels.")
    return example.ner_tags


class CRFNER:
    def __init__(self, show_progress: bool = True) -> None:
        self.show_progress = show_progress
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True,
        )

    def fit(self, train_examples: List[SentenceExample]) -> None:
        X_train = [
            sent2features(x)
            for x in tqdm(
                train_examples,
                desc="CRF features",
                disable=not self.show_progress,
            )
        ]
        y_train = [
            sent2labels(x)
            for x in tqdm(
                train_examples,
                desc="CRF labels",
                disable=not self.show_progress,
            )
        ]

        print("Fitting CRF...")
        self.model.fit(X_train, y_train)

    def predict(self, examples: List[SentenceExample]) -> List[List[str]]:
        X = [
            sent2features(x)
            for x in tqdm(
                examples,
                desc="CRF predict features",
                disable=not self.show_progress,
            )
        ]
        return self.model.predict(X)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
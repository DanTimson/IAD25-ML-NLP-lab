from __future__ import annotations

from typing import List, Tuple


def extract_entities(tags: List[str]) -> List[Tuple[str, int, int]]:
    entities = []
    start = None
    ent_type = None

    for i, tag in enumerate(tags):
        if tag == "O":
            if ent_type is not None:
                entities.append((ent_type, start, i - 1))
                start = None
                ent_type = None
            continue

        prefix, cur_type = tag.split("-", 1)

        if prefix == "B":
            if ent_type is not None:
                entities.append((ent_type, start, i - 1))
            start = i
            ent_type = cur_type

        elif prefix == "I":
            if ent_type is None or ent_type != cur_type:
                if ent_type is not None:
                    entities.append((ent_type, start, i - 1))
                start = i
                ent_type = cur_type

    if ent_type is not None:
        entities.append((ent_type, start, len(tags) - 1))

    return entities


def entity_level_prf1(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> dict:
    tp = 0
    fp = 0
    fn = 0

    for true_tags, pred_tags in zip(y_true, y_pred):
        true_ents = set(extract_entities(true_tags))
        pred_ents = set(extract_entities(pred_tags))

        tp += len(true_ents & pred_ents)
        fp += len(pred_ents - true_ents)
        fn += len(true_ents - pred_ents)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
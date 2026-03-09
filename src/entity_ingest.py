from __future__ import annotations

from typing import List, Tuple

from src.preprocess import SentenceExample
from src.entity import EntityStore
from tqdm.auto import tqdm


def spans_from_iob(tokens: List[str], tags: List[str]) -> List[Tuple[str, str, int, int]]:
    spans = []
    start = None
    ent_type = None

    for i, tag in enumerate(tags):
        if tag == "O":
            if ent_type is not None:
                text = " ".join(tokens[start:i])
                spans.append((text, ent_type, start, i - 1))
                start = None
                ent_type = None
            continue

        prefix, cur_type = tag.split("-", 1)

        if prefix == "B":
            if ent_type is not None:
                text = " ".join(tokens[start:i])
                spans.append((text, ent_type, start, i - 1))
            start = i
            ent_type = cur_type
        elif prefix == "I":
            if ent_type is None or ent_type != cur_type:
                if ent_type is not None:
                    text = " ".join(tokens[start:i])
                    spans.append((text, ent_type, start, i - 1))
                start = i
                ent_type = cur_type

    if ent_type is not None:
        text = " ".join(tokens[start:len(tokens)])
        spans.append((text, ent_type, start, len(tokens) - 1))

    return spans


def ingest_examples(
    store: EntityStore,
    examples: List[SentenceExample],
    predicted_tags: List[List[str]],
    show_progress: bool = True,
) -> None:
    for ex, tags in tqdm(
        list(zip(examples, predicted_tags)),
        total=len(examples),
        desc="Ingest entities",
        disable=not show_progress,
    ):
        sent_text = " ".join(ex.tokens)
        spans = spans_from_iob(ex.tokens, tags)

        for ent_text, ent_type, start, end in spans:
            store.add_mention(
                entity_name=ent_text,
                category=ent_type,
                text=sent_text,
                sentence_id=ex.sentence_id,
                start=start,
                end=end,
            )
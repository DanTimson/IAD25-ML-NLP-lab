from __future__ import annotations

from collections import Counter
from html import escape
from typing import Dict, List

import matplotlib.pyplot as plt
from wordcloud import WordCloud

from src.entity import EntityStore


ENTITY_COLORS = {
    "per": "#add8e6",
    "gpe": "#ffcccb",
    "geo": "#ffd580",
    "eve": "#d8bfd8",
    "nat": "#c7f2c7",
    "art": "#f7dc6f",
    "tim": "#d6eaf8",
    "org": "#a9dfbf",
}


def plot_category_wordcloud(store: EntityStore, category: str, save_path: str | None = None) -> None:
    words = []
    for ent in store.entities.values():
        if ent.category == category:
            words.extend([m["surface_form"] for m in ent.mentions])

    if not words:
        raise ValueError(f"No words found for category '{category}'.")

    freqs = Counter(words)
    wc = WordCloud(width=1200, height=600, background_color="white")
    img = wc.generate_from_frequencies(freqs)

    plt.figure(figsize=(14, 7))
    plt.imshow(img, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word cloud: {category}")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def highlight_entities_html(tokens: List[str], tags: List[str]) -> str:
    parts = []
    for token, tag in zip(tokens, tags):
        safe_token = escape(token)

        if tag == "O":
            parts.append(safe_token)
        else:
            _, ent_type = tag.split("-", 1)
            color = ENTITY_COLORS.get(ent_type, "#eeeeee")
            parts.append(
                f"<span style='background-color:{color}; padding:2px 4px; border-radius:4px;'>{safe_token} ({ent_type})</span>"
            )

    return " ".join(parts)
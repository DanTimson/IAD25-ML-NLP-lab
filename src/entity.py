from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Entity:
    canonical_name: str
    category: str
    description: str = ""
    aliases: set[str] = field(default_factory=set)
    mentions: List[dict] = field(default_factory=list)


class EntityStore:
    def __init__(self) -> None:
        self.entities: Dict[str, Entity] = {}
        self.categories: set[str] = set()

    @staticmethod
    def normalize(text: str) -> str:
        return " ".join(text.strip().lower().split())

    def add_category(self, category: str) -> None:
        self.categories.add(category)

    def add_entity(
        self,
        name: str,
        category: str,
        description: str = "",
        aliases: Optional[List[str]] = None,
    ) -> None:
        norm = self.normalize(name)
        self.categories.add(category)

        if norm not in self.entities:
            self.entities[norm] = Entity(
                canonical_name=name,
                category=category,
                description=description,
                aliases=set(),
            )

        if aliases:
            for alias in aliases:
                self.entities[norm].aliases.add(alias)

        if description and not self.entities[norm].description:
            self.entities[norm].description = description

    def delete_entity(self, name: str) -> bool:
        norm = self.normalize(name)
        if norm in self.entities:
            del self.entities[norm]
            return True
        return False

    def reassign_entity(self, name: str, new_category: str) -> bool:
        norm = self.normalize(name)
        if norm not in self.entities:
            return False
        self.entities[norm].category = new_category
        self.categories.add(new_category)
        return True

    def add_mention(
        self,
        entity_name: str,
        category: str,
        text: str,
        sentence_id: int,
        start: int,
        end: int,
    ) -> None:
        norm = self.normalize(entity_name)

        if norm not in self.entities:
            self.add_entity(entity_name, category)

        self.entities[norm].mentions.append({
            "text": text,
            "sentence_id": sentence_id,
            "start": start,
            "end": end,
            "surface_form": entity_name,
        })

    def get_related_texts(self, entity_name: str) -> List[str]:
        norm = self.normalize(entity_name)
        if norm not in self.entities:
            return []
        return [m["text"] for m in self.entities[norm].mentions]

    def get_entity_overview(self, entity_name: str) -> Optional[dict]:
        norm = self.normalize(entity_name)
        if norm not in self.entities:
            return None

        ent = self.entities[norm]
        return {
            "canonical_name": ent.canonical_name,
            "category": ent.category,
            "description": ent.description,
            "aliases": sorted(ent.aliases),
            "num_mentions": len(ent.mentions),
            "related_texts": [m["text"] for m in ent.mentions],
        }

    def explain_word(self, entity_name: str) -> str:
        info = self.get_entity_overview(entity_name)
        if info is None:
            return f"No information known about '{entity_name}'."

        if info["description"]:
            return info["description"]

        return (
            f"'{info['canonical_name']}' is known as category '{info['category']}' "
            f"with {info['num_mentions']} stored mention(s)."
        )

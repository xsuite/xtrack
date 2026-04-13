"""Utilities to annotate and collect categorized class APIs."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CategorySpec:
    name: str


class CategorizedAPICollector:
    def __init__(self, category_order: Iterable[str]):
        self.category_order = tuple(category_order)
        self._category_set = set(self.category_order)

    def ensure_known_category(self, category: str) -> None:
        if category not in self._category_set:
            raise ValueError(f"Unknown API category: {category!r}")

    def collect(self, cls):
        out = OrderedDict(
            (
                category,
                {
                    "name": category,
                    "methods": [],
                    "properties": [],
                },
            )
            for category in self.category_order
        )

        for name, member in cls.__dict__.items():
            if name.startswith("_"):
                continue

            if isinstance(member, property):
                category = getattr(member.fget, "__api_category__", None)
                if category is not None:
                    self.ensure_known_category(category)
                    out[category]["properties"].append(name)
                continue

            if isinstance(member, (staticmethod, classmethod)):
                func = member.__func__
                category = getattr(member, "__api_category__", None)
                if category is None:
                    category = getattr(func, "__api_category__", None)
            elif callable(member):
                category = getattr(member, "__api_category__", None)
            else:
                category = None

            if category is not None:
                self.ensure_known_category(category)
                out[category]["methods"].append(name)

        return [item for item in out.values() if item["methods"] or item["properties"]]

    def validate(self, cls, *, strict=False, ignore=()):
        ignore = set(ignore)

        categorized = set()
        for item in self.collect(cls):
            categorized.update(item["methods"])
            categorized.update(item["properties"])

        public = set()
        for name, member in cls.__dict__.items():
            if name.startswith("_") or name in ignore:
                continue
            if isinstance(member, property):
                public.add(name)
                continue
            if isinstance(member, (staticmethod, classmethod)):
                public.add(name)
                continue
            if callable(member):
                public.add(name)

        missing = sorted(public - categorized)
        if strict and missing:
            raise ValueError(
                f"Uncategorized public API in {cls.__name__}: {missing}"
            )
        return missing


def api_category(category: str):
    def decorator(obj):
        setattr(obj, "__api_category__", category)
        return obj

    return decorator


def property_with_category(category: str):
    def decorator(func):
        setattr(func, "__api_category__", category)
        return property(func)

    return decorator

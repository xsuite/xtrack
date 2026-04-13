"""Utilities to annotate and collect grouped class APIs."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class GroupSpec:
    name: str


class GroupedAPICollector:
    def __init__(self, group_order: Iterable[str]):
        self.group_order = tuple(group_order)
        self._group_set = set(self.group_order)

    def ensure_known_group(self, group: str) -> None:
        if group not in self._group_set:
            raise ValueError(f"Unknown doc group: {group!r}")

    def collect(self, cls):
        out = OrderedDict(
            (
                group,
                {
                    "name": group,
                    "methods": [],
                    "properties": [],
                },
            )
            for group in self.group_order
        )

        for name, member in cls.__dict__.items():
            if name.startswith("_"):
                continue

            if isinstance(member, property):
                group = getattr(member.fget, "__doc_group__", None)
                if group is not None:
                    self.ensure_known_group(group)
                    out[group]["properties"].append(name)
                continue

            if isinstance(member, (staticmethod, classmethod)):
                func = member.__func__
                group = getattr(member, "__doc_group__", None)
                if group is None:
                    group = getattr(func, "__doc_group__", None)
            elif callable(member):
                group = getattr(member, "__doc_group__", None)
            else:
                group = None

            if group is not None:
                self.ensure_known_group(group)
                out[group]["methods"].append(name)

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
                f"Ungrouped public API in {cls.__name__}: {missing}"
            )
        return missing


def doc_group(group: str):
    def decorator(obj):
        setattr(obj, "__doc_group__", group)
        return obj

    return decorator


def property_with_doc_group(group: str):
    def decorator(func):
        setattr(func, "__doc_group__", group)
        return property(func)

    return decorator

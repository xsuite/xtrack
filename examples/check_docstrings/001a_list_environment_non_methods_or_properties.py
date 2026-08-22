#!/usr/bin/env python3
"""List xtrack.Environment members that are neither methods nor properties."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path


# Ensure local package imports when running from this monorepo.
_THIS_FILE = Path(__file__).resolve()
_XTRACK_REPO_ROOT = _THIS_FILE.parents[2]
if str(_XTRACK_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_XTRACK_REPO_ROOT))

_XSUITE_PACKAGES_ROOT = _THIS_FILE.parents[3]
for _pkg_name in ("xobjects", "xdeps", "xpart", "xfields", "xtrack"):
    _pkg_root = _XSUITE_PACKAGES_ROOT / _pkg_name
    if _pkg_root.is_dir() and str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))


def _is_property(cls, name: str) -> bool:
    class_member = getattr(cls, name, None)
    return isinstance(class_member, property)


def _is_method(member) -> bool:
    return (
        inspect.ismethod(member)
        or inspect.isfunction(member)
        or inspect.isbuiltin(member)
        or inspect.ismethoddescriptor(member)
        or inspect.isroutine(member)
    )


def main():
    import xtrack as xt

    environment = xt.Environment()
    cls = type(environment)

    found = []
    skipped = []

    for name in dir(environment):
        try:
            member = getattr(environment, name)
        except Exception as exc:  # pragma: no cover - defensive introspection
            skipped.append((name, type(exc).__name__))
            continue

        if _is_property(cls, name):
            continue

        if _is_method(member):
            continue

        found.append((name, type(member).__name__))

    print(f"Instance inspected: {cls.__module__}.{cls.__name__}")
    print()
    print(f"Members in dir(environment) that are not methods/properties ({len(found)}):")
    for name, type_name in sorted(found):
        print(f"- {name} ({type_name})")

    if skipped:
        print()
        print(f"Skipped (failed getattr) ({len(skipped)}):")
        for name, exc_name in sorted(skipped):
            print(f"- {name} [{exc_name}]")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Report public xtrack.Line methods and properties missing docstrings."""

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


def _is_missing_doc(obj) -> bool:
    doc = inspect.getdoc(obj)
    return doc is None or not doc.strip()


def _public_methods_and_properties(cls):
    methods = []
    properties = []

    for name, member in cls.__dict__.items():
        if name.startswith("_"):
            continue

        if isinstance(member, property):
            properties.append((name, member))
            continue

        if isinstance(member, (staticmethod, classmethod)):
            methods.append((name, member.__func__))
            continue

        if inspect.isfunction(member):
            methods.append((name, member))

    return methods, properties


def main():
    import xtrack as xt
    cls = xt.Line
    methods, properties = _public_methods_and_properties(cls)

    missing_methods = sorted(name for name, member in methods if _is_missing_doc(member))
    missing_properties = sorted(name for name, member in properties if _is_missing_doc(member))

    print(f"Class inspected: {cls.__module__}.{cls.__name__}")
    print()

    print(f"Methods missing doc ({len(missing_methods)}):")
    for name in missing_methods:
        print(f"- [ ] {name}")

    print()

    print(f"Properties missing doc ({len(missing_properties)}):")
    for name in missing_properties:
        print(f"- [ ] {name}")


if __name__ == "__main__":
    main()

# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from pathlib import Path
from typing import Literal, TypeAlias, get_args

from xobjects.general import _print  # noqa: F401
import requests
import gzip

_pkg_root = Path(__file__).parent.absolute()

DEPRECATION_INFO_PREP_1_0 = (
    " This deprecation is part of the interface cleanup in view of "
    "the 1.0 release.")


class _LOC:
    def __init__(self, name=None):
        self.name = name

    def __repr__(self):
        return self.name


START = _LOC('START')
END = _LOC('END')

AnchorName: TypeAlias = Literal['start', 'center', 'centre', 'end']
ANCHOR_NAMES = get_args(AnchorName)


def parse_anchor_spec(
        element_name: str,
        *,
        default_anchor: AnchorName | None = None,
) -> tuple[str, AnchorName | None]:
    """Parse an ``element@anchor`` string.

    Parameters
    ----------
    element_name : str
        Element name, optionally followed by ``@start``, ``@center``,
        ``@centre``, or ``@end``.
    default_anchor : str, optional
        Anchor returned when ``spec`` does not contain ``@``.

    Returns
    -------
    element_name : str
        Element name without the anchor suffix.
    anchor : str or None
        Parsed anchor, or ``default_anchor`` when no anchor suffix is present.
    """
    if default_anchor is not None and default_anchor not in ANCHOR_NAMES:
        raise ValueError(f'Invalid default anchor `{default_anchor}`.')

    if '@' not in element_name:
        return element_name, default_anchor

    element_name, anchor = element_name.rsplit('@', 1)

    if not element_name:
        raise ValueError(f'Invalid anchored element specification `{element_name}`.')

    if anchor not in ANCHOR_NAMES:
        raise ValueError(
            f'Invalid anchor `{anchor}` in `{element_name}`. Allowed anchors are: {ANCHOR_NAMES}.'
        )

    return element_name, anchor

def read_url(url, timeout=0.1, binary=False):
    """
    Read content from a URL.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # Raise an error for bad responses
        if url.endswith('.gz'):
            out = gzip.decompress(response.content)
            if binary:
                return out
            else:
                return out.decode("utf-8")
        else:
            if binary:
                return response.content
            else:
                return response.text
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to read from URL {url}: {e}")

def _compare_versions(v1, v2):
    """Compare two version strings.

    Returns:
        -1 if v1 < v2
         0 if v1 == v2
         1 if v1 > v2
    """
    def parse_version(v):
        return [int(part) for part in v.split('.') if part.isdigit()]

    parts1 = parse_version(v1)
    parts2 = parse_version(v2)

    # Extend the shorter list with zeros (e.g., 1.0 vs 1.0.0)
    length = max(len(parts1), len(parts2))
    parts1.extend([0] * (length - len(parts1)))
    parts2.extend([0] * (length - len(parts2)))

    for p1, p2 in zip(parts1, parts2):
        if p1 < p2:
            return -1
        elif p1 > p2:
            return 1
    return 0

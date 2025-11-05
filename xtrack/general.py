# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from enum import Enum
from pathlib import Path
from xobjects.general import _print  # noqa: F401
import requests
import gzip

_pkg_root = Path(__file__).parent.absolute()


class _LOC:
    def __init__(self, name=None):
        self.name = name

    def __repr__(self):
        return self.name


START = _LOC('START')
END = _LOC('END')

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

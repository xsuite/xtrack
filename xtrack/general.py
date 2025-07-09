# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from enum import Enum
from pathlib import Path
from xobjects.general import _print  # noqa: F401
import requests

_pkg_root = Path(__file__).parent.absolute()


class _LOC:
    def __init__(self, name=None):
        self.name = name

    def __repr__(self):
        return self.name


START = _LOC('START')
END = _LOC('END')

def read_url(url, timeout=0.1):
    """
    Read content from a URL.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # Raise an error for bad responses
        return response.text
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to read from URL {url}: {e}")
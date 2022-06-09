# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict


_function = type(lambda: None)


def _pro_default(default):
    type_default = type(default)
    if type_default is _function:
        mut = default()
        if type(mut) is list:
            return List, field(default_factory=default)
        elif type(mut) is dict:
            return Dict, field(default_factory=default)
    elif type_default in [list, dict]:
        raise ValueError(f"Mutable default {default} not allowed")
    else:
        return type_default, default


class _MetaElement(type):
    def __new__(cls, clsname, bases, dct):
        description = dct.get("_description", [])
        extra = dct.get("_extra", [])
        dct["_base_fields"] = [dd[0] for dd in description]
        dct["_extra_fields"] = [dd[0] for dd in extra]
        dct["_fields"] = dct["_base_fields"] + dct["_extra_fields"]
        ann = {}
        dct["__annotations__"] = ann
        for name, unit, desc, default in description:
            ann[name], dct[name] = _pro_default(default)
        for name, unit, desc, default in extra:
            ann[name], dct[name] = _pro_default(default)
        try:
            doc = [dct["__doc__"], "\nFields:\n"]
        except KeyError:
            doc = ["\nFields:\n"]
        fields = [
            f"    - {name:10} [{unit+']:':5} {desc} "
            for name, unit, desc, default in description
        ]
        fields += [
            f"    - {name:10} [{unit+']:':5} {desc} "
            for name, unit, desc, default in extra
        ]
        doc += fields
        dct["__doc__"] = "\n".join(doc)
        newclass = super(_MetaElement, cls).__new__(cls, clsname, bases, dct)
        return dataclass(newclass)


class Base(metaclass=_MetaElement):

    iscollective = False

    def get_fields(self, keepextra=False):
        if keepextra:
            return self.__class__._fields
        else:
            return self.__class__._base_fields

    def to_dict(self, keepextra=False):
        out = {kk: getattr(self, kk) for kk in self.get_fields(keepextra)}
        out["__class__"] = self.__class__.__name__
        return out

    @classmethod
    def from_dict(cls, dct, keepextra=True):
        self = cls()
        for kk in cls._base_fields:
            setattr(self, kk, dct[kk])
        if keepextra:
            for kk in cls._extra_fields:
                if kk in dct:
                    setattr(self, kk, dct[kk])
        return self

    def copy(self, keepextra=True):
        return self.__class__.from_dict(self.to_dict(keepextra), keepextra)


class Element(Base):
    pass



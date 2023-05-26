# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2023.                 #
# ######################################### #

import abc
import re

from itertools import zip_longest
from typing import List, Tuple, Iterator


class ElementSlicingScheme(abc.ABC):
    def __init__(self, slicing_order: int):
        self.slicing_order = slicing_order

    @abc.abstractmethod
    def element_weights(self) -> List[float]:
        """Define a list of weights of length `self.slicing_order`, containing
         the weight of each element slice.
        """
        pass

    @abc.abstractmethod
    def drift_weights(self) -> List[float]:
        """Define a list of weights of length `self.slicing_order + 1`,
        containing the weight of each drift slice.
        """
        pass

    def __iter__(self) -> Iterator[Tuple[float, bool]]:
        """
        Give an iterator for weights of slices and, assuming the first slice is
        a drift, followed by an element slice, and so on.
        Returns
        -------
        Iterator[Tuple[float, bool]]
            Iterator of weights and whether the weight is for a drift.
        """
        for drift_weight, elem_weight in zip_longest(
                self.drift_weights(),
                self.element_weights(),
                fillvalue=None,
        ):
            yield drift_weight, True

            if elem_weight is None:
                break

            yield elem_weight, False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.slicing_order})"


class Uniform(ElementSlicingScheme):
    def element_weights(self):
        return [1. / self.slicing_order] * self.slicing_order

    def drift_weights(self):
        slices = self.slicing_order + 1
        return [1. / slices] * slices


class Teapot(ElementSlicingScheme):
    def element_weights(self):
        return [1. / self.slicing_order] * self.slicing_order

    def drift_weights(self):
        if self.slicing_order == 1:
            return [0.5, 0.5]

        edge_weight = 1. / (2 * (1 + self.slicing_order))
        middle_weight = self.slicing_order / (self.slicing_order ** 2 - 1)
        middle_weights = [middle_weight] * (self.slicing_order - 1)

        return [edge_weight, *middle_weights, edge_weight]


class Strategy:
    def __init__(self, slicing, name=None, element_type=None):
        if name is not None and isinstance(name, str):
            self.name_regex = re.compile(name)
        else:
            self.name_regex = None

        self.element_type = element_type
        self.slicing = slicing

    def _match_on_name(self, name):
        if self.name_regex is None:
            return True
        return self.name_regex.match(name)

    def _match_on_type(self, element):
        if self.element_type is None:
            return True
        return isinstance(element, self.element_type)

    def match_element(self, name, element):
        return self._match_on_name(name) and self._match_on_type(element)

    def __repr__(self):
        params = {
            'slicing': self.slicing,
            'element_type': self.element_type,
            'name': self.name_regex.pattern if self.name_regex else None,
        }
        formatted_params = ', '.join(
            f'{kk}={vv!r}' for kk, vv in params.items() if vv is not None
        )
        return f"{type(self).__name__}({formatted_params})"

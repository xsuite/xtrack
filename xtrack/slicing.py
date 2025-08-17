# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2023.                 #
# ######################################### #

import abc
import re
from abc import ABC

from itertools import zip_longest
from typing import List, Tuple, Iterator, Optional, Literal

import numpy as np

from .progress_indicator import progress

import xtrack as xt

APER_ELEMS_REGEX = re.compile(r'(.*)_aper(_(tilt|offset)_(entry|exit))?')


class ElementSlicingScheme(abc.ABC):
    def __init__(
            self,
            slicing_order: int,
            mode: Literal['thin', 'thick'] = 'thin',
    ):
        if slicing_order < 1:
            raise ValueError("A slicing scheme must have at least one slice.")

        self.mode = mode

        if mode == 'thick':
            # In thick mode we count the number of thick slices, 'drifts',
            # not thin ones, so we need to decrement the slicing order.
            slicing_order -= 1

        self.slicing_order = slicing_order

    @abc.abstractmethod
    def element_weights(self, element_length: float) -> List[float]:
        """Define a list of weights of length `self.slicing_order`, containing
         the weight of each element slice.
        """

    @abc.abstractmethod
    def drift_weights(self, element_length: float) -> List[float]:
        """Define a list of weights of length `self.slicing_order + 1`,
        containing the weight of each drift slice.
        """

    def iter_weights(
            self,
            element_length: float = None,
    ) -> Iterator[Tuple[float, bool]]:
        """
        Give an iterator for weights of slices and, assuming the first slice is
        a drift, followed by an element slice, and so on.
        Returns
        -------
        Iterator[Tuple[float, bool]]
            Iterator of weights and whether the weight is for a drift.
        """
        for drift_weight, elem_weight in zip_longest(
                self.drift_weights(element_length),
                self.element_weights(element_length),
                fillvalue=None,
        ):
            yield drift_weight, True

            if elem_weight is None:
                break

            if self.mode == 'thin':
                yield elem_weight, False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.slicing_order})"


class Uniform(ElementSlicingScheme):
    def element_weights(self, element_length=None):
        if self.slicing_order == 0 and self.mode == 'thick':
            return [1.]

        return [1. / self.slicing_order] * self.slicing_order

    def drift_weights(self, element_length=None):
        if self.slicing_order == 0:
            return [1.]
        slices = self.slicing_order + 1
        return [1. / slices] * slices


class Teapot(ElementSlicingScheme):
    def element_weights(self, element_length=None):
        if self.slicing_order == 0 and self.mode == 'thick':
            return [1.]

        return [1. / self.slicing_order] * self.slicing_order

    def drift_weights(self, element_length=None):

        if self.slicing_order == 0:
            return [1.]

        if self.slicing_order == 1:
            return [0.5, 0.5]

        edge_weight = 1. / (2 * (1 + self.slicing_order))
        middle_weight = self.slicing_order / (self.slicing_order ** 2 - 1)
        middle_weights = [middle_weight] * (self.slicing_order - 1)

        return [edge_weight, *middle_weights, edge_weight]


class Custom(ElementSlicingScheme):
    """The custom slicing scheme slices the element at the fixed s coordinates.

    Arguments
    ---------
    at_s
        The s values at which the elements should be sliced. The beginning of
        the element is assumed to be at zero.
    mode:
        Thick or thin slicing.
    """
    def __init__(
            self,
            at_s: List[float],
            mode: Literal['thin', 'thick'] = 'thick',
    ):
        slicing_order = len(at_s)
        if mode == 'thick':
            # In thick number of slices is one more than cuts
            slicing_order = len(at_s) + 1

        super().__init__(slicing_order, mode)

        # Precompute the known weights (all but the last slice, which depends
        # on the weight of the element being sliced)
        self.at_s = np.array([0.] + at_s)
        self.section_lengths = self.at_s[1:] - self.at_s[:-1]

    def element_weights(self, element_length=None):
        return [1. / self.slicing_order] * self.slicing_order

    def drift_weights(self, element_length: float) -> List[float]:
        section_lengths = np.concatenate([
            self.section_lengths,
            [element_length - self.at_s[-1]],
        ])
        weights = section_lengths / element_length
        return weights.tolist()

    def __repr__(self):
        return f'Custom(at_s={self.at_s.tolist()})'


class Strategy:
    def __init__(self, slicing, name=None, element_type=None, exact=False):
        if name is None and element_type is None:
            exact = True

        self.regex = not exact

        if name is not None and isinstance(name, str) and self.regex:
            self.match_name = re.compile(name)
        else:
            self.match_name = name

        self.element_type = element_type
        self.slicing = slicing

    def _match_on_name(self, name):
        if self.regex:
            return self.match_name.match(name)
        return self.match_name == name

    def _match_on_type(self, element, line):
        if isinstance(element, xt.Replica):
            element = element.resolve(line)
        return isinstance(element, self.element_type)

    def match_element(self, name, element, line):

        if isinstance(element, xt.Drift):
            matched = False
            if self.match_name and not self.element_type:
                matched = self._match_on_name(name)
            elif self.element_type and not self.match_name:
                matched = self._match_on_type(element, line)
            elif self.match_name and self.element_type:
                matched = self._match_on_name(name) and self._match_on_type(element, line)
        else:
            matched = True
            if self.match_name:
                matched = matched and self._match_on_name(name)
            if self.element_type:
                matched = matched and self._match_on_type(element, line)
        return matched

    def __repr__(self):
        params = {
            'slicing': self.slicing,
            'element_type': self.element_type,
            'name': self.match_name.pattern if self.regex else self.match_name,
        }
        formatted_params = ', '.join(
            f'{kk}={vv!r}' for kk, vv in params.items() if vv is not None
        )
        return f"{type(self).__name__}({formatted_params})"


class Slicer:
    def __init__(self, line, slicing_strategies: List[Strategy]):
        """
        An object that slices a line in place according to a list of slicing
        strategies.

        Parameters
        ----------
        line : Line
            The line to slice.
        slicing_strategies : List[Strategy]
            A list of slicing strategies to apply to the line.
        """

        self._line = line
        self._slicing_strategies = [xt.Strategy(None, element_type=xt.Drift) # Do nothing to drifts by default
                                    ] + slicing_strategies
        self._has_expressions = line.vars is not None

    def slice_in_place(self, _edge_markers=True):

        self._line._frozen_check()

        thin_names = []

        tt = self._line.get_table()
        assert tt.name[-1] == '_end_point'
        tt = tt.rows[:-1]
        slices = {}
        for ii, nn in enumerate(progress(tt.name, desc='Slicing line')):

            enn=tt.env_name[ii]
            element = self._line.element_dict[enn]

            subsequence = [enn]
            # Don't slice already thin elements and drifts
            if (not xt.line._is_thick(element, self._line)
                or (hasattr(element, 'length') and element.length == 0)):
                pass
            else:
                chosen_slicing = self._scheme_for_element(element, nn, self._line)
                if chosen_slicing is not None:
                    subsequence = self._slice_element(
                        enn, element, _edge_markers=_edge_markers,
                        chosen_slicing=chosen_slicing)
                if subsequence is None:
                    subsequence = [enn]

            thin_names += subsequence
            slices[nn] = subsequence

        # Commit the changes to the line
        self._line.element_names = thin_names

        return slices

    def _slice_element(self, name, element, chosen_slicing, _edge_markers=True) -> Optional[List[str]]:
        """Slice element and return slice names, or None if no slicing."""

        if isinstance(element, xt.Drift) or type(element).__name__.startswith('DriftSlice'):
            _edge_markers = False

        # Make the slices and add them to line.element_dict (so far inactive)
        slices_to_add = self._make_slices(
            element=element,
            chosen_slicing=chosen_slicing,
            name=name,
        )

        if _edge_markers:

            if isinstance(element, xt.Replica):
                element = element.resolve(self._line)
            _buffer = element._buffer

            entry_marker, exit_marker = f'{name}_entry', f'{name}_exit'
            self._line.element_dict[entry_marker] = xt.Marker(_buffer=_buffer)
            self._line.element_dict[exit_marker] = xt.Marker(_buffer=_buffer)
            slices_to_add = [entry_marker] + slices_to_add + [exit_marker]

        # Handle aperture
        if isinstance(element, xt.Replica):
            element = element.resolve(self._line)
        if (hasattr(element, 'name_associated_aperture')
            and element.name_associated_aperture is not None):
            new_slices_to_add = []
            aper_index = 0
            for nn in slices_to_add:
                ee = self._line.element_dict[nn]
                if (type(ee).__name__.startswith('ThinSlice')
                    or type(ee).__name__.startswith('ThickSlice')):
                    aper_name = f'{name}_aper..{aper_index}'
                    self._line.element_dict[aper_name] = xt.Replica(
                        parent_name=element.name_associated_aperture)
                    new_slices_to_add += [aper_name]
                    aper_index += 1
                new_slices_to_add += [nn]
            slices_to_add = new_slices_to_add

        return slices_to_add

    def _scheme_for_element(self, element, name, line):
        """Choose a slicing strategy for the element"""

        slicing_found = False
        chosen_slicing = None

        for strategy in reversed(self._slicing_strategies):
            if strategy.match_element(name, element, line):
                slicing_found = True
                chosen_slicing = strategy.slicing
                break
        if not slicing_found:
            raise ValueError(f'No slicing strategy found for the element '
                             f'{name}: {element}.')
        return chosen_slicing

    def _make_slices(self, element, chosen_slicing, name):
        """
        Add the slices to the line.element_dict. If the element has expressions
        then the expressions will be added to the slices.

        Parameters
        ----------
        element : BeamElement
            A thick element to slice.
        chosen_slicing : ElementSlicingScheme
            The slicing scheme to use for the element.
        name : str
            The name of the element.

        Returns
        -------
        list
            A list of the names of the slices that were added.
        """

        parent_name = name
        if isinstance(element, xt.Replica):
            parent_name = element.resolve(self._line, get_name=True)
            element = self._line[parent_name]

        drift_idx, element_idx = 0, 0
        slices_to_append = []

        if hasattr(element, '_entry_slice_class'):
            nn = f'{name}..entry_map'
            if nn in self._line.element_dict:
                i_entry = 0
                while (nn := f'{name}..entry_map_{i_entry}') in self._line.element_dict:
                    i_entry += 1
            ee = element._entry_slice_class(
                    _parent=element, _buffer=element._buffer)
            ee.parent_name = parent_name
            self._line.element_dict[nn] = ee
            slices_to_append.append(nn)

        if not hasattr(element, 'length'):
            # Slicing a thick slice of a another element
            assert hasattr(element, '_parent')
            assert element.isthick
            elem_length = element._parent.length * element.weight
            elem_weight = element.weight
            slice_parent_name = element.parent_name
            slice_parent = element._parent
            isdriftslice = type(element).__name__.startswith('DriftSlice')
        else:
            elem_length = element.length
            elem_weight = 1.
            slice_parent_name = parent_name
            slice_parent = element
            isdriftslice = False

        if chosen_slicing.mode == 'thin' or isdriftslice:
            for weight, is_drift in chosen_slicing.iter_weights(elem_length):
                prename = "" if isdriftslice else "drift_"
                if is_drift:
                    while (nn := f'{prename}{name}..{drift_idx}') in self._line.element_dict:
                        drift_idx += 1
                    ee = slice_parent._drift_slice_class(
                            _parent=slice_parent, _buffer=element._buffer,
                            weight=weight * elem_weight)
                    ee.parent_name = slice_parent_name
                    self._line.element_dict[nn] = ee
                    slices_to_append.append(nn)
                else:
                    if isdriftslice:
                        continue
                    while (nn := f'{name}..{element_idx}') in self._line.element_dict:
                        element_idx += 1
                    if slice_parent._thin_slice_class is not None:
                        ee = slice_parent._thin_slice_class(
                                _parent=slice_parent, _buffer=element._buffer,
                                weight=weight * elem_weight)
                        ee.parent_name = slice_parent_name
                        self._line.element_dict[nn] = ee
                        slices_to_append.append(nn)
        elif chosen_slicing.mode == 'thick':
            for weight, is_drift in chosen_slicing.iter_weights(elem_length):
                while (nn := f'{name}..{element_idx}') in self._line.element_dict:
                    element_idx += 1
                ee = slice_parent._thick_slice_class(
                        _parent=slice_parent, _buffer=element._buffer,
                        weight=weight * elem_weight)
                ee.parent_name = slice_parent_name
                self._line.element_dict[nn] = ee
                slices_to_append.append(nn)
        else:
            raise ValueError(f'Unknown slicing mode: {chosen_slicing.mode}')

        if hasattr(element, '_exit_slice_class'):
            nn = f'{name}..exit_map'
            if nn in self._line.element_dict:
                i_exit = 0
                while (nn := f'{name}..exit_map_{i_exit}') in self._line.element_dict:
                    i_exit += 1
            ee = element._exit_slice_class(
                    _parent=element, _buffer=element._buffer)
            ee.parent_name = parent_name
            self._line.element_dict[nn] = ee
            slices_to_append.append(nn)

        slice_parent._movable = True # Force movable
        element._movable = True # Force movable

        return slices_to_append

    def _make_copies(self, element_names, index):
        new_names = []
        for element_name in element_names:
            element = self._line.element_dict[element_name]
            new_element_name = f'{element_name}..{index}'
            self._line.element_dict[new_element_name] = element.copy()
            new_names.append(new_element_name)

        return new_names

    def _order_set_by_line(self, set_to_order: set):
        """Order a set of element names by their order in the line."""
        assert isinstance(set_to_order, (set, frozenset))
        try:
            return sorted(set_to_order, key=self._line.element_names.index)
        except:
            return set_to_order

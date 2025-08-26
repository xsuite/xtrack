# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from typing import Tuple
import numpy as np

import xobjects as xo
import xtrack as xt

from .general import _print

from xobjects.struct import Struct, MetaStruct

from .line import Line, mk_class_namespace, _has_backtrack


class SerializationHeader(xo.Struct):
    """
    In a predetermined place in the buffer we have the metadata
    offset and the element type names. These have to be separate,
    because in order to rebuild TrackerData we need to first build
    ElementRefClass.
    """
    element_ref_data_offset = xo.UInt64
    reftype_names = xo.String[:]


class TrackerData:

    def __init__(
            self,
            element_dict,
            element_names,
            element_s_locations,
            line_length,
            cache=None,
            kernel_element_classes=None,
            extra_element_classes=(),
            allow_move=False,
            _context=None,
            _buffer=None,
            _offset=None,
            _no_resolve_parents=False,
    ):
        """
        Create an immutable line suitable for serialisation.

        Parameters
        ----------
        element_dict : dict
            Dictionary of elements, keyed by name.
        element_names : list
            List of element names.
        element_s_locations : list
            List of element s locations.
        line_length : float
            Length of the line.
        kernel_element_classes : list, optional
            Explicit list of classes of elements of the line; if `None`,
            will be inferred from list.
        extra_element_classes : tuple, optional
            If `kernel_element_classes` is `None`, this list will be used to augment
            the inferred list of element classes.
        """
        if _offset is not None:
            raise ValueError('`_offset` is not supported yet')

        self._element_dict = element_dict.copy()

        # Handle replicas
        for nn in list(self._element_dict.keys()):
            ee = self._element_dict[nn]
            if isinstance(ee, xt.Replica):
                self._element_dict[nn] = ee.resolve(self._element_dict)

        self._element_names = tuple(element_names)
        self._elements = tuple([self._element_dict[ee] for ee in element_names])
        self._is_backtrackable = np.all([_has_backtrack(ee, element_dict)
                                         for ee in self._elements])
        self.extra_element_classes = extra_element_classes

        # If no buffer given, try to guess it from elements, if there is no
        # common buffer, try to guess the context from elements, if there is
        # no common context, a default will be taken.
        if _buffer is None:
            common_buffer = self.common_buffer_for_elements()
            if common_buffer is not None and _context in [common_buffer.context, None]:
                _buffer = common_buffer
            if _buffer is None and _context is None:
                _context = self.common_context_for_elements()
            _buffer = _buffer or xo.get_a_buffer(context=_context, size=64)
        elif _context is not None and _buffer.context is not _context:
            raise ValueError('The given context and buffer are not compatible.')

        check_passed = self.check_elements_in_common_buffer(_buffer, allow_move=allow_move)
        if not check_passed:
            raise RuntimeError('The elements are not in the same buffer')

        line_element_classes = set()
        for ee in self._elements:
            if ee._XoStruct in line_element_classes:
                continue
            line_element_classes.add(ee._XoStruct)
            if hasattr(ee, '_drift_slice_class') and ee._drift_slice_class:
                line_element_classes.add(ee._drift_slice_class._XoStruct)
            if hasattr(ee, '_thick_slice_class') and ee._thick_slice_class:
                line_element_classes.add(ee._thick_slice_class._XoStruct)
            if hasattr(ee, '_thin_slice_class') and ee._thin_slice_class:
                line_element_classes.add(ee._thin_slice_class._XoStruct)
            if hasattr(ee, '_entry_slice_class') and ee._entry_slice_class:
                line_element_classes.add(ee._entry_slice_class._XoStruct)
            if hasattr(ee, '_exit_slice_class') and ee._exit_slice_class:
                line_element_classes.add(ee._exit_slice_class._XoStruct)

        self.line_element_classes = line_element_classes
        self.element_s_locations = tuple(element_s_locations)
        self.line_length = line_length
        if cache is None:
            cache = {}
        self.cache = cache

        if not kernel_element_classes:
            kernel_element_classes = (
                line_element_classes | set(extra_element_classes))
            kernel_element_classes = sorted(kernel_element_classes, key=lambda cc: cc.__name__)
        else:
            if not line_element_classes.issubset(set(kernel_element_classes)):
                raise RuntimeError(
                    f'The following classes are not in `kernel_element_classes`: '
                    f'{line_element_classes - set(kernel_element_classes)}')

        ElementRefDataClass = xt.tracker._element_ref_data_class_from_element_classes(
                                            kernel_element_classes)
        self._element_ref_data = self.build_ref_data(_buffer, ElementRefDataClass)

        # Resolve slice parents
        for nn in element_names:
            if _no_resolve_parents:
                break
            if hasattr(self._element_dict[nn], '_parent'):
                this_parent = self._element_dict[
                    self._element_dict[nn].parent_name]
                this_parent._movable = True # Force movable
                if this_parent._buffer is not self._element_dict[nn]._buffer:
                    this_parent.move(_buffer=self._element_dict[nn]._buffer)
                self._element_dict[nn]._parent = this_parent
                this_parent._movable = True
                assert self._element_dict[nn]._parent._offset == self._element_dict[nn]._xobject._parent._offset

    def common_buffer_for_elements(self):
        """If all `self.elements` elements are in the same buffer,
        returns said buffer, otherwise returns `None`."""
        common_buffer = None
        for ee in self._elements:
            if hasattr(ee, '_buffer'):
                if common_buffer is None:
                    common_buffer = ee._buffer

                if ee._buffer is not common_buffer:
                    return None

        return common_buffer

    def common_context_for_elements(self):
        """If all `self.elements` elements are on the same context,
        returns said context, otherwise returns `None`."""
        common_context = None
        for ee in self._elements:
            if hasattr(ee, '_context'):
                if common_context is None:
                    common_context = ee._context

                if ee._context is not common_context:
                    return None

        return common_context

    def to_binary(self, buffer=None) -> Tuple[xo.context.XBuffer, int]:
        """
        Return a buffer containing a binary representation of the LineFrozen,
        together with the offset to the header in the buffer.
        These two are sufficient for recreating the line.
        """

        raise NotImplementedError('This method is not supported anymore')


    def check_elements_in_common_buffer(self, buffer, allow_move=False):
        """
        Move all the elements to the common buffer, if they are not already
        there.
        """
        for nn, ee in self._element_dict.items():
            if not hasattr(ee, '_buffer'):
                continue
            if ee._buffer is not buffer:
                if allow_move:
                    ee.move(_buffer=buffer)
                else:
                    return False

        return True

    def build_ref_data(self, buffer, element_ref_data_class):
        """
        Ensure all the elements of the line are in the buffer (which will be
        created if `buffer` is equal to `None`), and write the line metadata
        to it. If the buffer is empty, the metadata will be at the beginning.
        Returns the metadata xobject.
        """

        element_ref_data = element_ref_data_class(
            elements=len(self._elements),
            names=list(self._element_names),
            _buffer=buffer,
        )

        element_ref_data.elements = [
            self._element_dict[name]._xobject for name in self.element_names
        ]

        return element_ref_data

    @classmethod
    def from_binary(
        cls,
        buffer: xo.context.XBuffer,
        header_offset: int,
        extra_element_classes: tuple = (),
    ):
        raise NotImplementedError('This method is not supported anymore')

    @property
    def elements(self):
        return self._elements

    @property
    def element_names(self):
        return self._element_names

    @property
    def _buffer(self):
        return self._element_ref_data._buffer

    @property
    def _offset(self):
        return self._element_ref_data._offset

    @property
    def _context(self):
        return self._element_ref_data._context

    def __getstate__(self):
        out = self.__dict__.copy()
        out['_element_ref_data'] = (
            self._element_ref_data._buffer, self._element_ref_data._offset)
        out['kernel_element_classes'] = [cc._DressingClass for cc in self.kernel_element_classes]
        out['line_element_classes'] = [cc._DressingClass for cc in self.line_element_classes]
        out['extra_element_classes'] = [cc._DressingClass for cc in self.extra_element_classes]
        return out

    def __setstate__(self, state):
        buffer, offset = state.pop('_element_ref_data')
        kernel_element_classes = state.pop('kernel_element_classes')
        kernel_element_classes = [cc._XoStruct for cc in kernel_element_classes]
        self.__dict__.update(state)
        self.line_element_classes = [cc._XoStruct for cc in self.line_element_classes]
        self.extra_element_classes = [cc._XoStruct for cc in self.extra_element_classes]

        element_refs_cls = xt.tracker._element_ref_data_class_from_element_classes(
                                            kernel_element_classes)
        self._element_ref_data = element_refs_cls._from_buffer(
            buffer=buffer,
            offset=offset,
        )

    @property
    def kernel_element_classes(self):
        return self._element_ref_data.elements._itemtype._reftypes


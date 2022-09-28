# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xobjects as xo

from .line import Line
from . import beam_elements


class LineFrozen:
    def __init__(self, line, _context=None, _buffer=None,  _offset=None):
        self.line = line

        if _buffer is None:
            common_buffer = self.common_buffer_for_elements()
            if common_buffer is not None and common_buffer.context is _context:
                _buffer = common_buffer

        num_elements = len(line.element_names)

        element_data_types = set(ee._XoStruct for ee in line.elements)
        sorted_element_data_types = sorted(
            element_data_types, key=lambda cc: cc.__name__)

        class ElementRefClass(xo.UnionRef):
            _reftypes = sorted_element_data_types

        LineDataClass = ElementRefClass[num_elements]

        line_data = LineDataClass(
           _context=_context,
           _buffer=_buffer,
           _offset=_offset)

        for ii, ee in enumerate(line.elements):
            assert hasattr(ee, '_XoStruct')  # is already xobject
            if ee._buffer != line_data._buffer:
                ee.move(_buffer=line_data._buffer)

            line_data[ii] = ee._xobject

        # freeze line
        line.element_names = tuple(line.element_names)

        self.element_s_locations = tuple(line.get_s_elements())
        self._line_data = line_data
        self._LineDataClass = LineDataClass
        self._ElementRefClass = ElementRefClass

    def common_buffer_for_elements(self):
        """If all `self.line` elements are in the same buffer,
        returns said buffer, otherwise returns `None`."""
        common_buffer = None
        for ee in self.line.elements:
            if hasattr(ee, '_buffer'):
                if common_buffer is None:
                    common_buffer = ee._buffer

                if ee._buffer is not common_buffer:
                    return None

        return common_buffer

    def serialize(self, context=xo.context_default):
        buffer = context.new_buffer(0)

        # As the first element in the buffer we have element type names.
        # These have to be separate, because in order to rebuild LineData
        # we need to first build ElementRefClass.
        reftype_names = xo.String[:](
            [
                reftype._DressingClass.__name__
                for reftype in self._ElementRefClass._reftypes
            ],
            _buffer=buffer,
        )

        # Then, we write the actual line metadata
        class LineData(xo.Struct):
            elements = self._ElementRefClass[:]
            names = xo.String[:]

        line_data = LineData(
            elements=self._line_data,
            names=list(self.line.element_names),
            _buffer=buffer,
        )

        return buffer

    @classmethod
    def deserialize(cls, buffer):
        reftype_names = xo.String[:]._from_buffer(buffer, 0)
        reftypes = [
            getattr(beam_elements, reftype)._XoStruct
            for reftype in reftype_names
        ]

        # With the reftypes loaded we can create our classes
        class ElementRefClass(xo.UnionRef):
            _reftypes = reftypes

        class LineData(xo.Struct):
            elements = ElementRefClass[:]
            names = xo.String[:]

        # Read the line data
        start_offset = reftype_names._get_size()
        line_data = LineData._from_buffer(buffer, start_offset)

        # Recreate and redress line elements
        hybrid_cls_for_struct = {
            getattr(beam_elements, reftype)._XoStruct:
                getattr(beam_elements, reftype) for reftype in reftype_names
        }
        elements = [
            hybrid_cls_for_struct[elem.__class__](
                _xobject=elem,
                _buffer=buffer,
            ) for elem in line_data.elements
        ]

        line = Line(
            elements=elements,
            element_names=line_data.names,
        )
        return LineFrozen(line=line)

    @property
    def elements(self):
        return self.line.elements

    @property
    def element_names(self):
        return self.line.element_names

    @property
    def _buffer(self):
        return self._line_data._buffer

    @property
    def _offset(self):
        return self._line_data._offset

# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xobjects as xo

from .line import Line
from . import beam_elements


class LineFrozen:
    class SerializationHeader(xo.Struct):
        """
        In a predetermined place in the buffer we have the metadata
        offset and the element type names. These have to be separate,
        because in order to rebuild LineData we need to first build
        ElementRefClass.
        """
        metadata_start = xo.UInt64
        reftype_names = xo.String[:]

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

        self._tracker_data = self.build_tracker_data(line_data._buffer)

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

    def serialize(self, context=xo.context_default) \
            -> xo.context.XBuffer:
        """
        Create a buffer containing a binary representation of the LineFrozen,
        from which it can be recreated quickly.
        """
        existing_buffer = self._line_data._buffer
        tracker_data = self.build_tracker_data(buffer=existing_buffer)
        target_buffer = context.new_buffer(0)
        # Put the pointer to the metadata at the beginning
        header = self.build_header(
            buffer=target_buffer,
            metadata_start=tracker_data._offset,
        )
        # Expand the buffer to allow the copy
        target_buffer.grow(header._size + existing_buffer.capacity)
        # Follow the header by the contents of the existing buffer
        target_buffer.update_from_xbuffer(
            offset=header._size,
            source=existing_buffer,
            source_offset=0,
            nbytes=existing_buffer.capacity,
        )
        return target_buffer

    def build_header(self, buffer, metadata_start) -> SerializationHeader:
        """
        Build a serialization header in the buffer. This should be in a
        predetermined location, as the data is necessary for decoding
        the line metadata.
        """
        return self.SerializationHeader(
            metadata_start=metadata_start,
            reftype_names=[
                reftype._DressingClass.__name__
                for reftype in self._ElementRefClass._reftypes
            ],
            _buffer=buffer,
        )

    def build_tracker_data(self, buffer):
        """
        Ensure all the elements of the line are in the buffer, and write
        the line metadata to it. If the buffer is empty, the metadata will
        be at the beginning. Returns the metadata xobject.
        """
        class TrackerData(xo.Struct):
            elements = self._ElementRefClass[:]
            names = xo.String[:]

        tracker_data = TrackerData(
            elements=len(self.line.elements),
            names=list(self.line.element_names),
            _buffer=buffer,
        )

        # Move all the elements into buffer, so they don't get duplicated.
        # We only do it now, as we need to make sure line_data is already
        # allocated after reftype_names.
        moved_element_dict = {}
        for name, elem in self.line.element_dict.items():
            if elem._buffer is not buffer:
                moved_element_dict[name] = elem._XoStruct(
                    elem._xobject, _buffer=buffer
                )
            else:
                moved_element_dict[name] = elem._xobject

        tracker_data.elements = [
            moved_element_dict[name] for name in self.line.element_names
        ]

        return tracker_data

    @classmethod
    def deserialize(cls, buffer: xo.context.XBuffer) -> 'LineFrozen':
        header = cls.SerializationHeader._from_buffer(buffer, 0)
        reftypes = [
            getattr(beam_elements, reftype)._XoStruct
            for reftype in header.reftype_names
        ]

        # With the reftypes loaded we can create our classes
        class ElementRefClass(xo.UnionRef):
            _reftypes = reftypes

        class TrackerData(xo.Struct):
            elements = ElementRefClass[:]
            names = xo.String[:]

        # Read the line data
        start_offset = header._size

        # Since the offset is relative to the first position after the
        # header, we need to shift the buffer. This is done to avoid
        # copying the buffer into the new one.
        # TODO: This is hacky solution, and needs to be improved (XView?)
        shifted_buffer = buffer.context.new_buffer(0)
        shifted_buffer.buffer = buffer.buffer[start_offset:]
        shifted_buffer.capacity = buffer.capacity - start_offset
        shifted_buffer.chunks = []  # mark whole buffer as allocated,
                                    # it is editable but any previous
                                    # free space is lost forever

        # We can now load the line from the shifted buffer
        line_data = TrackerData._from_buffer(
            buffer=shifted_buffer,
            offset=int(header.metadata_start)
        )

        # Recreate and redress line elements
        hybrid_cls_for_struct = {
            getattr(beam_elements, reftype)._XoStruct:
                getattr(beam_elements, reftype)
            for reftype in header.reftype_names
        }

        element_dict = {}
        for ii, elem in enumerate(line_data.elements):
            name = line_data.names[ii]
            if name in element_dict:
                continue

            hybrid_cls = hybrid_cls_for_struct[elem.__class__]
            element_dict[name] = hybrid_cls(_xobject=elem, _buffer=shifted_buffer)

        line = Line(
            elements=element_dict,
            element_names=line_data.names,
        )
        line_frozen = LineFrozen(line=line)
        line_frozen._ElementRefClass = ElementRefClass

        return line_frozen

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

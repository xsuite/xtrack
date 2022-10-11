# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from collections import OrderedDict
from typing import Tuple

import xobjects as xo

from .line import Line
from . import beam_elements


class LineFrozen:
    class SerializationHeader(xo.Struct):
        """
        In a predetermined place in the buffer we have the metadata
        offset and the element type names. These have to be separate,
        because in order to rebuild TrackerData we need to first build
        ElementRefClass.
        """
        metadata_start = xo.UInt64
        reftype_names = xo.String[:]

    @staticmethod
    def tracker_data_factory(element_ref_class) -> 'TrackerData':
        class TrackerData(xo.Struct):
            elements = element_ref_class[:]
            names = xo.String[:]

        return TrackerData

    def __init__(
            self,
            line,
            element_classes=None,
            extra_element_classes=[],
            _context=None,
            _buffer=None,
            _offset=None,
    ):
        """
        Create an immutable line suitable for serialisation.

        :param xt.Line line: a line
        :param List[xo.Struct] element_classes: explicit list of classes of
            elements of the line; if `None`, will be inferred from list.
        :param List[xo.Struct] extra_element_classes: if `element_classes`
            is `None`, this list will be used to augment the inferred list
            of element classes.
        """
        self.line = line

        if _buffer is None:
            common_buffer = self.common_buffer_for_elements()
            if common_buffer is not None and _context in [common_buffer.context, None]:
                _buffer = common_buffer
            _buffer = _buffer or xo.get_a_buffer(context=_context, size=0)

        num_elements = len(line.element_names)

        if not element_classes:
            element_classes = set(ee._XoStruct for ee in line.elements)
            element_classes |= set(extra_element_classes)
            element_classes = sorted(element_classes, key=lambda cc: cc.__name__)
        self.element_classes = element_classes

        class ElementRefClass(xo.UnionRef):
            _reftypes = self.element_classes

        # freeze line
        line.element_names = tuple(line.element_names)
        self.element_s_locations = tuple(line.get_s_elements())
        self._ElementRefClass = ElementRefClass
        self._tracker_data = self.build_tracker_data(_buffer)

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

    def serialize(self, buffer=None) -> Tuple[xo.context.XBuffer, int]:
        """
        Return a buffer containing a binary representation of the LineFrozen,
        together with the offset to the header in the buffer.
        These two are sufficient for recreating the line.
        """
        _tracker_data = self._tracker_data
        if not buffer:
            buffer = _tracker_data._buffer

        if buffer is not _tracker_data._buffer:
            _tracker_data = self.build_tracker_data(buffer)

        header = self.build_header(
            buffer=buffer,
            metadata_start=_tracker_data._offset,
        )

        return buffer, header._offset

    def build_header(self, buffer, metadata_start) -> SerializationHeader:
        """
        Build a serialization header in the buffer. This contains all
        the necessary for decoding the line metadata.
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
        Ensure all the elements of the line are in the buffer (which will be
        created if `buffer` is equal to `None`), and write the line metadata
        to it. If the buffer is empty, the metadata will be at the beginning.
        Returns the metadata xobject.
        """
        tracker_data_cls = self.tracker_data_factory(self._ElementRefClass)

        tracker_data = tracker_data_cls(
            elements=len(self.line.elements),
            names=list(self.line.element_names),
            _buffer=buffer,
        )

        # Move all the elements into buffer, so they don't get duplicated.
        # We only do it now, as we need to make sure tracker_data is already
        # allocated after reftype_names.
        moved_element_dict = {}
        for name, elem in self.line.element_dict.items():
            if elem._buffer is not buffer:
                elem.move(_buffer=buffer)
            moved_element_dict[name] = elem._xobject

        tracker_data.elements = [
            moved_element_dict[name] for name in self.line.element_names
        ]

        return tracker_data

    @classmethod
    def deserialize(
        cls,
        buffer: xo.context.XBuffer,
        header_offset: int,
        extra_element_classes=[],
    ) -> 'LineFrozen':
        header = cls.SerializationHeader._from_buffer(
            buffer=buffer,
            offset=header_offset,
        )

        extra_classes_dict = {
            elem_cls.__name__: elem_cls
            for elem_cls in extra_element_classes
        }

        element_hybrid_classes = []
        for reftype in header.reftype_names:
            if reftype in extra_classes_dict:
                element_hybrid_classes.append(extra_classes_dict[reftype])
            elif hasattr(beam_elements, reftype):
                element_hybrid_classes.append(getattr(beam_elements, reftype))
            else:
                ValueError(f"Cannot find the type `{reftype}`")

        # With the reftypes loaded we can create our classes
        element_classes = [elem._XoStruct for elem in element_hybrid_classes]

        class ElementRefClass(xo.UnionRef):
            _reftypes = element_classes

        # We can now load the line from the buffer
        tracker_data_cls = cls.tracker_data_factory(ElementRefClass)
        tracker_data = tracker_data_cls._from_buffer(
            buffer=buffer,
            offset=int(header.metadata_start)
        )

        # Recreate and redress line elements
        hybrid_cls_for_xstruct = {
            elem._XoStruct: elem for elem in element_hybrid_classes
        }

        element_dict = {}
        for ii, elem in enumerate(tracker_data.elements):
            name = tracker_data.names[ii]
            if name in element_dict:
                continue

            hybrid_cls = hybrid_cls_for_xstruct[elem.__class__]
            element_dict[name] = hybrid_cls(_xobject=elem, _buffer=buffer)

        line = Line(
            elements=element_dict,
            element_names=tracker_data.names,
        )
        line_frozen = LineFrozen(line=line, element_classes=element_classes)
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
        return self._tracker_data._buffer

    @property
    def _offset(self):
        return self._tracker_data._offset

    @property
    def _context(self):
        return self._tracker_data._context

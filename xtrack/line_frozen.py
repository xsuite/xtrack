import xobjects as xo

from . import beam_elements as be

def seq_typename_to_xtclass(typename, external_elements):
    if typename in external_elements.keys():
        return external_elements[typename]
    else:
        return getattr(be, typename)

class LineFrozen:
    def __init__(self, line,
           _context=None, _buffer=None,  _offset=None):

        num_elements = len(line.element_names)

        element_data_types = set(ee.XoStruct for ee in line.elements)
        sorted_element_data_types = sorted(
			element_data_types, key = lambda cc:cc.__name__)

        class ElementRefClass(xo.UnionRef):
            _reftypes = sorted_element_data_types

        LineDataClass = ElementRefClass[num_elements]

        line_data = LineDataClass(
            _context=_context,
            _buffer=_buffer,
             _offset=_offset)

        for ii, (ee, nn) in enumerate(zip(line.elements,
                                      line.element_names)):
            assert hasattr(ee, 'XoStruct') # is already xobject
            if ee._buffer != line_data._buffer:
                ee._move_to(_buffer=line_data._buffer)

            line_data[ii] = ee._xobject

        # freeze line
        line.element_names = tuple(line.element_names)

        self.line = line

        self.element_s_locations = tuple(line.get_s_elements())
        self._line_data = line_data
        self._LineDataClass = LineDataClass
        self._ElementRefClass = ElementRefClass

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

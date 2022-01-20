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


        num_elements = len(line.elements)


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

        assert len(line.elements) == len(line.element_names)

        elements = []
        element_names = []

        for ii, (ee, nn) in enumerate(zip(line.elements,
                                      line.element_names)):
            assert hasattr(ee, 'XoStruct') # is already xobject
            if ee._buffer != line_data._buffer:
                ee._move_to(_buffer=line_data._buffer)

            elements.append(ee)
            element_names.append(nn)
            line_data[ii] = ee._xobject

        self.elements = tuple(elements)
        self.element_names = tuple(element_names)

        line.elements = self.elements
        line.element_names = self.element_names

        self.element_s_locations = tuple(line.get_s_elements())
        self._line_data = line_data
        self._LineDataClass = LineDataClass
        self._ElementRefClass = ElementRefClass

    @property
    def _buffer(self):
        return self._line_data._buffer

    @property
    def _offset(self):
        return self._line_data._offset

import xobjects as xo

from . import beam_elements as be

class Line():
    def __init__(self, sequence,
           _context=None, _buffer=None,  _offset=None):

        num_elements = len(sequence.elements)
        elem_type_names = set([ee.__class__.__name__
                                for ee in sequence.elements])
        element_types = [getattr(be, nn) for nn in elem_type_names]
        element_data_types = [cc.XoStruct for cc in element_types]

        ElementRefClass = xo.Ref(*element_data_types)
        LineDataClass = ElementRefClass[num_elements]
        line_data = LineDataClass(_context=_context,
                _buffer=_buffer, _offset=_offset)
        elements = []
        for ii, ee in enumerate(sequence.elements):
            XtClass = getattr(be, ee.__class__.__name__)
            xt_ee = XtClass(_buffer=line_data._buffer, **ee.to_dict())
            elements.append(xt_ee)
            line_data[ii] = xt_ee._xobject

        self.elements = tuple(elements)
        self._line_data = line_data
        self._LineDataClass = LineDataClass
        self._ElementRefClass = ElementRefClass

    @property
    def _buffer(self):
        return self._line_data._buffer

    @property
    def _offset(self):
        return self._line_data._offset

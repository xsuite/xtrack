import xobjects as xo

from . import beam_elements as be

def get_pysixtrack_xfields_mapping():
    try:
        import xfields as xf # I need to do it like this to avoid circular import
        xfields_elements = {
            'BeamBeam4D': xf.BeamBeamBiGaussian2D,
            'BeamBeam6D': xf.BeamBeamBiGaussian3D}
    except ImportError:
        print('Xfields not available')
        xfields_elements = {
            'BeamBeam4D': None,
            'BeamBeam6D': None}
    return xfields_elements

def seq_typename_to_xtclass(typename, external_elements):
    if typename in external_elements.keys():
        return external_elements[typename]
    else:
        return getattr(be, typename)

class Line():
    def __init__(self, sequence,
           _context=None, _buffer=None,  _offset=None, external_elements=None,
           use_xfields_elements=True):

        '''
        At the moment the sequence is assumed to be a pysixtrack line.
        This will be generalized in the future.
        '''

        if external_elements is None:
            external_elements = {}

        if use_xfields_elements:
            external_elements.update(get_pysixtrack_xfields_mapping())

        num_elements = len(sequence.elements)
        elem_type_names = set([ee.__class__.__name__
                                for ee in sequence.elements])

        # Identify xtrack element classes
        element_data_types = []
        for nn in sorted(elem_type_names):
            cc = seq_typename_to_xtclass(nn, external_elements)
            element_data_types.append(cc.XoStruct)

        ElementRefClass = xo.Ref(*element_data_types)
        LineDataClass = ElementRefClass[num_elements]
        line_data = LineDataClass(_context=_context,
                _buffer=_buffer, _offset=_offset)
        elements = []
        for ii, ee in enumerate(sequence.elements):
            XtClass = seq_typename_to_xtclass(ee.__class__.__name__, external_elements)
            if hasattr(XtClass, 'from_pysixtrack'):
                xt_ee = XtClass.from_pysixtrack(ee, _buffer=line_data._buffer)
            else:
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

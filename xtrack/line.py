import xobjects as xo

from . import beam_elements as be

def get_xline_xfields_mapping():
    try:
        import xfields as xf # I need to do it like this to avoid circular import
        xfields_elements = {
            'BeamBeam4D': xf.BeamBeamBiGaussian2D,
            'BeamBeam6D': xf.BeamBeamBiGaussian3D,
            'SCQGaussProfile': xf.SpaceChargeBiGaussian}
    except ImportError:
        print('Xfields not available')
        xfields_elements = {
            'BeamBeam4D': None,
            'BeamBeam6D': None,
            'SCQGaussProfile': None}
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
        At the moment the sequence is assumed to be a xline line.
        This will be generalized in the future.
        '''

        if external_elements is None:
            external_elements = {}

        if use_xfields_elements:
            external_elements.update(get_xline_xfields_mapping())

        num_elements = len(sequence.elements)

        # Identify element types that are not xobjects 
        elem_type_names = set([ee.__class__.__name__
                                for ee in sequence.elements
                                if not hasattr(ee, 'XoStruct')])
        element_data_types = []
        for nn in sorted(elem_type_names):
            cc = seq_typename_to_xtclass(nn, external_elements)
            element_data_types.append(cc.XoStruct)

        # Identify element types that are xobjects
        for ee in sequence.elements:
            if (hasattr(ee, 'XoStruct')
                    and ee.XoStruct not in element_data_types):
                element_data_types.append(ee.XoStruct)


        class ElementRefClass(xo.UnionRef):
            _reftypes=element_data_types

        LineDataClass = ElementRefClass[num_elements]
        line_data = LineDataClass(_context=_context,
                _buffer=_buffer, _offset=_offset)
        elements = []
        for ii, ee in enumerate(sequence.elements):
            if hasattr(ee, 'XoStruct'): # is already xobject
                assert ee._buffer == line_data._buffer, (
                        'Copy from different buffer not yet implemented')
                xt_ee = ee
            else: # needs to be converted
                XtClass = seq_typename_to_xtclass(ee.__class__.__name__, external_elements)
                if hasattr(XtClass, 'from_xline'):
                    xt_ee = XtClass.from_xline(ee, _buffer=line_data._buffer)
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

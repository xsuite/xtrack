import numpy as np

import sixtracktools
import pysixtrack

six = sixtracktools.SixInput(".")
pyst_line = pysixtrack.Line.from_sixinput(six)


import xtrack as xt
import xobjects as xo
xtelems = []

buff = xo.ContextCpu().new_buffer()
#buff = xo.ContextCupy().new_buffer()


class Line():
    def __init__(self, sequence,
           _context=None, _buffer=None,  _offset=None):

        num_elements = len(sequence.elements)
        elem_type_names = set([ee.__class__.__name__
                                for ee in sequence.elements])
        element_types = [getattr(xt, nn) for nn in elem_type_names]
        element_data_types = [cc.XoStruct for cc in element_types]

        ElementRefClass = xo.Ref(*element_data_types)
        LineDataClass = ElementRefClass[num_elements]
        line_data = LineDataClass(_context=_context,
                _buffer=_buffer, _offset=_offset)
        elements = []
        for ii, ee in enumerate(sequence.elements):
            XtClass = getattr(xt, ee.__class__.__name__)
            xt_ee = XtClass(_buffer=line_data._buffer, **ee.to_dict())
            elements.append(xt_ee)
            line_data[ii] = xt_ee._xobject

        self.elements = tuple(elements)
        self._line_data = line_data
        self._LineDataClass = LineDataClass
        self._ElementRefClass = ElementRefClass

print('Creating line...')
xtline = Line(_buffer=buff, sequence=pyst_line)

# Check the test
# ixtelems[2].knl[0]*=1.00000001

print('Performing check...')
for ie, (ee, xtee) in enumerate(zip(pyst_line.elements, xtline.elements)):
    print(f'{ie}/{len(pyst_line.elements)}   ', end='\r',flush=True)
    dd = ee.to_dict()
    for kk in dd.keys():
        if kk == '__class__':
            continue
        if hasattr(dd[kk], '__iter__'):
            for ii in range(len(dd[kk])):
                assert np.isclose(dd[kk][ii], getattr(xtee, kk)[ii],
                        rtol=1e-10, atol=1e-14)
        else:
            assert np.isclose(dd[kk], getattr(xtee, kk),
                    rtol=1e-10, atol=1e-14)


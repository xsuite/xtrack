import numpy as np

import sixtracktools
import pysixtrack

six = sixtracktools.SixInput(".")
line = pysixtrack.Line.from_sixinput(six)

elem_types = set([ee.__class__ for ee in line.elements])

import xtrack as xt
import xobjects as xo
xtelems = []

buff = xo.ContextCpu().new_buffer()
buff = xo.ContextCupy().new_buffer()

xtelem_types = [getattr(xt, cc.__name__) for cc in elem_types]

class Line():
    def __init__(self, element_types=(), num_elements=0,
            _buffer=None, _context=None, _offset=None):

        element_data_types = [cc.XoStruct for cc in element_types]
        ElementRefClass = xo.Ref(*element_data_types)
        self._elements_data = ElementRefClass[num_elements](
                _buffer=_buffer, _offset=_offset)

    @property
    def _buffer(self):
        return self._elements_data._buffer

xtline = Line(element_types=xtelem_types,
        num_elements=len(line.elements), _buffer=buff)

print('Creating line...')
for ii, ee in enumerate(line.elements):
    xtclass = getattr(xt, ee.__class__.__name__)
    xtee = xtclass(_buffer=xtline._buffer, **ee.to_dict())
    xtelems.append(xtee)
    xtline._elements_data[ii] = xtee._xobject

@property
def _buffer(self):
    return self._elements_data._buffer
# Chenk the test
# ixtelems[2].knl[0]*=1.00000001
print('Performing check...')
for ie, (ee, xtee) in enumerate(zip(line.elements, xtelems)):
    print(f'{ie}/{len(line.elements)}   ', end='\r',flush=True)
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


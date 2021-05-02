import numpy as np

import xtrack as xt
import xobjects as xo
import sixtracktools
import pysixtrack

context = xo.ContextCpu()

six = sixtracktools.SixInput(".")
pyst_line = pysixtrack.Line.from_sixinput(six)
sixdump = sixtracktools.SixDump101("res/dump3.dat")
part0_pyst = pysixtrack.Particles(**sixdump[0::2][0].get_minimal_beam())
part1_pyst = pysixtrack.Particles(**sixdump[1::2][0].get_minimal_beam())

print('Creating line...')
xtline = xt.Line(_context=context, sequence=pyst_line)

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


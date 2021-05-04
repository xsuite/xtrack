import numpy as np
from scipy.special import factorial

import xtrack as xt
import xobjects as xo
import sixtracktools
import pysixtrack

context = xo.ContextCpu()

six = sixtracktools.SixInput(".")
pyst_line = pysixtrack.Line.from_sixinput(six)
sixdump = sixtracktools.SixDump101("res/dump3.dat")

# TODO: The two particles look identical, to be checked
part0_pyst = pysixtrack.Particles(**sixdump[0::2][0].get_minimal_beam())
part1_pyst = pysixtrack.Particles(**sixdump[1::2][0].get_minimal_beam())

particles = xt.Particles(pysixtrack_particles=[part0_pyst, part1_pyst])

print('Creating line...')
xtline = xt.Line(_context=context, sequence=pyst_line)

print('Build capi')
sources = []
kernels = {}
cdefs = []
for cc in xtline._ElementRefClass._rtypes:
    ss, kk, dd = cc._gen_c_api()
    sources.append(ss)
    kernels.update(kk)
    cdefs.append(dd)

context.add_kernels(sources, kernels, extra_cdef='\n\n'.join(cdefs))

# Check the test
# ixtelems[2].knl[0]*=1.00000001

print('Performing check...')
for ie, (ee, xtee) in enumerate(zip(pyst_line.elements, xtline.elements)):
    print(f'{ie}/{len(pyst_line.elements)}   ', end='\r',flush=True)
    dd = ee.to_dict()
    xtdatatype = xtee._xobject.__class__.__name__
    for kk in dd.keys():
        if kk == '__class__':
            continue
        if kk == 'knl':
            tmpmethod = getattr(context.kernels, xtdatatype+'_get_bal')
            cmethod = lambda obj, i0 : (tmpmethod(obj=obj, i0=2*i0)
                                          * factorial(i0, exact=True))
        elif kk == 'ksl':
            tmpmethod = getattr(context.kernels, xtdatatype+'_get_bal')
            cmethod = lambda obj, i0 : (tmpmethod(obj=obj, i0=2*i0+1)
                                          * factorial(i0, exact=True))
        elif kk == 'angle':
            tmpmethodsin = getattr(context.kernels, xtdatatype+'_get_sin_z')
            tmpmethodcos = getattr(context.kernels, xtdatatype+'_get_cos_z')
            cmethod = lambda obj: 180 / np.pi *np.arctan2(tmpmethodsin(obj=obj),
                                                          tmpmethodcos(obj=obj))
        else:
            cmethod = getattr(context.kernels, xtdatatype+'_get_'+kk)
        if hasattr(dd[kk], '__iter__'):
            for ii in range(len(dd[kk])):
                pyval = getattr(xtee, kk)[ii]
                assert np.isclose(dd[kk][ii], pyval,
                        rtol=1e-10, atol=1e-14)
                cval = cmethod(obj=xtee, i0=ii)
                assert np.isclose(dd[kk][ii], cval,
                        rtol=1e-10, atol=1e-14)
        else:
            pyval = getattr(xtee, kk)
            assert np.isclose(dd[kk], pyval,
                    rtol=1e-10, atol=1e-14)
            cval = cmethod(obj=xtee)
            assert np.isclose(dd[kk], cval,
                    rtol=1e-10, atol=1e-14)


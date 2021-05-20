from pathlib import Path

import numpy as np
import xobjects as xo
from scipy.special import factorial

from .dress import dress
from .particles import ParticlesData

thisfolder = Path(__file__).parent.absolute()
pkg_root = thisfolder

class DriftData(xo.Struct):
    length = xo.Float64
with open(thisfolder.joinpath('track_functions/drift.h')) as fid:
    DriftData.track_function_source = fid.read()
DriftData.track_kernel_source = '''
            /*gpukern*/
            void Drift_track_particles(
                             DriftData el,
                             ParticlesData particles){
            LocalParticle lpart;
            int64_t part_id = 0;                    //only_for_context cpu_serial cpu_openmp
            int64_t part_id = blockDim.x * blockIdx.x + threadIdx.x; //only_for_context cuda
            int64_t part_id = get_global_id(0);                    //only_for_context opencl

            int64_t n_part = ParticlesData_get_num_particles(particles);
            if (part_id<n_part){
                Particles_to_LocalParticle(particles, &lpart, part_id);
                Drift_track_local_particle(el, &lpart);
                }
            }
'''
DriftData.track_kernel_description = {'Drift_track_particles':
        xo.Kernel(args=[xo.Arg(DriftData, name='el'),
                        xo.Arg(ParticlesData, name='particles')])}

class Drift(dress(DriftData)):
    '''The drift...'''
    pass

class CavityData(xo.Struct):
    voltage = xo.Float64
    frequency = xo.Float64
    lag = xo.Float64
with open(thisfolder.joinpath('track_functions/cavity.h')) as fid:
    CavityData.track_function_source = fid.read()

class Cavity(dress(CavityData)):
    pass

class XYShiftData(xo.Struct):
    dx = xo.Float64
    dy = xo.Float64
with open(thisfolder.joinpath('track_functions/xyshift.h')) as fid:
    XYShiftData.track_function_source = fid.read()

class XYShift(dress(XYShiftData)):
    pass

class SRotationData(xo.Struct):
    cos_z = xo.Float64
    sin_z = xo.Float64
with open(thisfolder.joinpath('track_functions/srotation.h')) as fid:
    SRotationData.track_function_source = fid.read()

class SRotation(dress(SRotationData)):

    def __init__(self, angle=0, **nargs):
        anglerad = angle / 180 * np.pi
        nargs['cos_z']=np.cos(anglerad)
        nargs['sin_z']=np.sin(anglerad)
        super().__init__(**nargs)

    @property
    def angle(self):
        return np.arctan2(self.sin_z, self.cos_z) * (180.0 / np.pi)

class MultipoleData(xo.Struct):
    order = xo.Int64
    length = xo.Float64
    hxl = xo.Float64
    hyl = xo.Float64
    bal = xo.Float64[:]
with open(thisfolder.joinpath('track_functions/multipole.h')) as fid:
    MultipoleData.track_function_source = fid.read()

class Multipole(dress(MultipoleData)):

    def __init__(self, order=None, knl=None, ksl=None, bal=None, **kwargs):

        if bal is None and (
            knl is not None or ksl is not None or order is not None
        ):
            if knl is None:
                knl = []
            if ksl is None:
                ksl = []
            if order is None:
                order = 0

            n = max((order + 1), max(len(knl), len(ksl)))
            assert n > 0

            _knl = np.array(knl)
            nknl = np.zeros(n, dtype=_knl.dtype)
            nknl[: len(knl)] = knl
            knl = nknl
            del _knl
            assert len(knl) == n

            _ksl = np.array(ksl)
            nksl = np.zeros(n, dtype=_ksl.dtype)
            nksl[: len(ksl)] = ksl
            ksl = nksl
            del _ksl
            assert len(ksl) == n

            order = n - 1
            bal = np.zeros(2 * order + 2)

            idx = np.array([ii for ii in range(0, len(knl))])
            inv_factorial = 1.0 / factorial(idx, exact=True)
            bal[0::2] = knl * inv_factorial
            bal[1::2] = ksl * inv_factorial

            kwargs["bal"] = bal
            kwargs["order"] = order

        elif bal is not None and len(bal) > 0:
            kwargs["bal"] = bal
            kwargs["order"] = (len(bal) - 2) // 2


        # TODO: Remove when xobjects is fixed
        kwargs["bal"] = list(kwargs['bal'])
        self._temp_bal_length = len(kwargs['bal'])

        self.xoinitialize(**kwargs)

    @property
    def knl(self):
        idxes = np.array([ii for ii in range(0, self._temp_bal_length, 2)])
        return [self.bal[idx] * factorial(idx // 2, exact=True) for idx in idxes]

    @property
    def ksl(self):
        idxes = np.array([ii for ii in range(0, self._temp_bal_length, 2)])
        return [self.bal[idx + 1] * factorial(idx // 2, exact=True) for idx in idxes]
        #idx = np.array([ii for ii in range(0, len(self.bal), 2)])
        #return self.bal[idx + 1] * factorial(idx // 2, exact=True)




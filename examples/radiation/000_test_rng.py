import numpy as np

import xobjects as xo
import xtrack as xt
from pathlib import Path

ctx = xo.ContextCpu()
ctx = xo.ContextCupy()

part = xt.Particles(_context=ctx, p0c=6.5e12, x=[1,2,3])
part._init_random_number_generator()

class TestElement(xt.BeamElement):
     _xofields={
        'dummy': xo.Float64,
        }
TestElement.XoStruct.extra_sources = [
    xt._pkg_root.joinpath('random_number_generator/rng_src/base_rng.h'),
    xt._pkg_root.joinpath('random_number_generator/rng_src/local_particle_rng.h'),
    ]
TestElement.XoStruct.extra_sources.append('''
/*gpufun*/
void TestElement_track_local_particle(TestElementData el, LocalParticle* part0){
    //start_per_particle_block (part0->part)
        double rr = LocalParticle_generate_random_double(part);
        LocalParticle_set_x(part, rr);
    //end_per_particle_block
}
''')

telem = TestElement(_context=ctx)

telem.track(part)

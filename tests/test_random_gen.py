import numpy as np

import xobjects as xo
import xtrack as xt
import xline as xl
from pathlib import Path
def test_random_generation():
    for ctx in xo.context.get_test_contexts():
        print(f'{ctx}')

        part = xt.Particles(_context=ctx, p0c=6.5e12, x=[1,2,3])
        part._init_random_number_generator()

        class TestElement(xt.BeamElement):
             _xofields={
                'dummy': xo.Float64,
                }
        TestElement.XoStruct.extra_sources = [
            xt._pkg_root.joinpath(
                'random_number_generator/rng_src/base_rng.h'),
            xt._pkg_root.joinpath(
                'random_number_generator/rng_src/local_particle_rng.h'),
            ]
        TestElement.XoStruct.extra_sources.append('''
        /*gpufun*/
        void TestElement_track_local_particle(
                  TestElementData el, LocalParticle* part0){
            //start_per_particle_block (part0->part)
                double rr = LocalParticle_generate_random_double(part);
                LocalParticle_set_x(part, rr);
            //end_per_particle_block
        }
        ''')

        telem = TestElement(_context=ctx)

        telem.track(part)

        # Use turn-by turin monitor to acquire some statistics

        tracker = xt.Tracker(_buffer=telem._buffer,
                sequence=xl.Line(elements=[telem],
                    element_names='test_element'),
                    save_source_as='source.c')

        tracker.track(part, num_turns=1e6, turn_by_turn_monitor=True)

        import matplotlib.pyplot as plt
        plt.close('all')
        for i_part in range(part._capacity):
            x = tracker.record_last_track.x[i_part, :]
            assert np.all(x>0)
            assert np.all(x<1)
            hstgm, bin_edges = np.histogram(x,  bins=50, range=(0, 1), density=True)
            assert np.allclose(hstgm, 1, rtol=1e-10, atol=0.03)


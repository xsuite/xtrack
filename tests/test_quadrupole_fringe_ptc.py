# Checks entrance fringe for very strong quadrupole against values of PTC

import xtrack as xt
import numpy as np
from cpymad.madx import Madx

def test_quadrupole_fringe_ptc():
    b2 = 100
    b1 = 0
    length=1e-20

    x0 = 0.07
    px0 = 0.03
    y0 = 0.08
    py0 = 0.06

    # XSuite
    quadrupole = xt.Bend(length=length, k0=b1, k1=b2, edge_entry_model='1', edge_exit_model='1')  # 1 with quadrupole fringe
    line = xt.Line(elements=[ quadrupole])

    p0 = xt.Particles(x=x0,px=px0,y=y0,py=py0)

    line.discard_tracker()
    line.build_tracker()
    line.track(p0)
    
    mat = line.compute_one_turn_matrix_finite_differences(p0)['R_matrix']
    det = np.linalg.det(mat)
    
    assert np.isclose(det, 1.0)

    madx_sequence = line.to_madx_sequence('quadrupole_fringes')

    madx = Madx()
    madx.beam()
    madx.input(madx_sequence)
    madx.use('quadrupole_fringes')
    
    madx.input(f"""
               ptc_create_universe;
               ptc_create_layout, exact=true;
               ptc_setswitch, fringe=true;
               ptc_start, x={x0}, px={px0}, y={y0}, py={py0}, t=0.0, pt=0.0;
               ptc_track, TURNS=1;
               ptc_track_end;
               ptc_end;
               stop;
               """
               )
    
    df = madx.table.tracksumm.dframe()
    
    assert np.isclose(p0.x, df.x[-1])
    assert np.isclose(p0.px, df.px[-1])
    assert np.isclose(p0.y, df.y[-1])
    assert np.isclose(p0.py, df.py[-1])

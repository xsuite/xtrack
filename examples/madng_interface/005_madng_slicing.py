import xtrack as xt
import xobjects as xo
import numpy as np

line = xt.load('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
tw = line.twiss4d()

twng = line.madng_twiss()

line.cut_at_s(np.linspace(0, line.get_length(), 5000))
tw_sliced = line.twiss4d()
twng_sliced = line.madng_twiss()
tt_sliced = line.get_table()

assert np.all(np.array(sorted(list(set(tt_sliced.element_type)))) ==
    ['',
    'Cavity',
    'Drift',
    'DriftSlice',
    'Marker',
    'Multipole',
    'Octupole',
    'Quadrupole',
    'RBend',
    'Sextupole',
    'ThickSliceBend',
    'ThickSliceCavity',
    'ThickSliceMultipole',
    'ThickSliceOctupole',
    'ThickSliceQuadrupole',
    'ThickSliceRBend',
    'ThickSliceSextupole',
    'ThickSliceUniformSolenoid',
    'ThinSliceBendEntry',
    'ThinSliceBendExit',
    'ThinSliceOctupoleEntry',
    'ThinSliceOctupoleExit',
    'ThinSliceQuadrupoleEntry',
    'ThinSliceQuadrupoleExit',
    'ThinSliceRBendEntry',
    'ThinSliceRBendExit',
    'ThinSliceSextupoleEntry',
    'ThinSliceSextupoleExit',
    'ThinSliceUniformSolenoidEntry',
    'ThinSliceUniformSolenoidExit',
    'UniformSolenoid'])

twng_ip = twng.rows['ip.*']
twng_ip_sliced = twng_sliced.rows['ip.*']
xo.assert_allclose(twng_ip.s, twng_ip_sliced.s, rtol=1e-8)
xo.assert_allclose(twng_ip.beta11_ng, twng_ip_sliced.beta11_ng, rtol=1e-3)
xo.assert_allclose(twng_ip.beta22_ng, twng_ip_sliced.beta22_ng, rtol=1e-3)
xo.assert_allclose(twng_ip.wx_ng, twng_ip_sliced.wx_ng, rtol=1e-3)
xo.assert_allclose(twng_ip.wy_ng, twng_ip_sliced.wy_ng, rtol=1e-3)
xo.assert_allclose(twng_ip.dx_ng, twng_ip_sliced.dx_ng, atol=1e-6)
xo.assert_allclose(twng_ip.dy_ng, twng_ip_sliced.dy_ng, atol=1e-6)


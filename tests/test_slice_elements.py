import itertools
import numpy as np
import pytest

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

assert_allclose = xo.assert_allclose

@pytest.mark.parametrize('bend_type', [xt.Bend, xt.RBend])
@for_all_test_contexts
def test_thin_slice_bend(test_context, bend_type):

    bend = bend_type(k0=0.4, h=0.3, length=1,
                    edge_entry_angle=0.05, edge_entry_hgap=0.06, edge_entry_fint=0.08,
                    edge_exit_angle=0.05, edge_exit_hgap=0.06, edge_exit_fint=0.08)

    line = xt.Line(elements=[bend])

    line.configure_bend_model(edge='linear', core='expanded')

    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Teapot(10000))])
    line.build_tracker(_context=test_context)
    line._line_before_slicing.build_tracker(_context=test_context)
    assert line['e0..995'].parent_name == 'e0'
    assert line['e0..995']._parent is line.element_dict['e0']
    assert line['drift_e0..995'].parent_name == 'e0'
    assert line['drift_e0..995']._parent is line.element_dict['e0']
    assert line['e0..entry_map'].parent_name == 'e0'
    assert line['e0..entry_map']._parent is line.element_dict['e0']
    assert line['e0..exit_map'].parent_name == 'e0'
    assert line['e0..exit_map']._parent is line.element_dict['e0']

    p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03
                      ,_context=test_context)
    p_ref = p0.copy()
    p_slice = p0.copy()

    line.track(p_slice)
    line._line_before_slicing.track(p_ref)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-10)

    thin_slice_cls, drift_slice_type_cls, slice_entry_cls, slice_exit_cls = {
        xt.Bend: (
            xt.ThinSliceBend,
            xt.DriftSliceBend,
            xt.ThinSliceBendEntry,
            xt.ThinSliceBendExit,
        ),
        xt.RBend: (
            xt.ThinSliceRBend,
            xt.DriftSliceRBend,
            xt.ThinSliceRBendEntry,
            xt.ThinSliceRBendExit
        ),
    }[bend_type]

    line.to_json('ttt_thin_bend.json')
    line2 = xt.load('ttt_thin_bend.json')
    assert isinstance(line2['e0..995'], thin_slice_cls)
    assert line2['e0..995'].parent_name == 'e0'
    assert line2['e0..995']._parent is None
    assert line2['drift_e0..995'].parent_name == 'e0'
    assert line2['drift_e0..995']._parent is None
    assert line2['e0..entry_map'].parent_name == 'e0'
    assert line2['e0..entry_map']._parent is None
    assert line2['e0..exit_map'].parent_name == 'e0'
    assert line2['e0..exit_map']._parent is None

    line2.build_tracker(_context=test_context)
    assert isinstance(line2['e0..995'], thin_slice_cls)
    assert line2['e0..995'].parent_name == 'e0'
    assert line2['e0..995']._parent is line2.element_dict['e0']
    assert isinstance(line2['drift_e0..995'], drift_slice_type_cls)
    assert line2['drift_e0..995'].parent_name == 'e0'
    assert line2['drift_e0..995']._parent is line2.element_dict['e0']
    assert isinstance(line2['e0..entry_map'], slice_entry_cls)
    assert line2['e0..entry_map'].parent_name == 'e0'
    assert line2['e0..entry_map']._parent is line2.element_dict['e0']
    assert isinstance(line2['e0..exit_map'], slice_exit_cls)
    assert line2['e0..exit_map'].parent_name == 'e0'
    assert line2['e0..exit_map']._parent is line2.element_dict['e0']

    line.track(p_slice, backtrack=True)

    assert (p_slice.state == 1).all()
    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-10)

    line.optimize_for_tracking()

    assert isinstance(line['e0..995'], xt.SimpleThinBend)
    assert isinstance(line['drift_e0..995'], xt.Drift)

    assert isinstance(line['e0..entry_map'], xt.DipoleEdge)
    assert isinstance(line['e0..exit_map'], xt.DipoleEdge)

    p_slice = p0.copy()
    line.track(p_slice)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-10)

    line.track(p_slice, backtrack=True)

    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-10)

@for_all_test_contexts
def test_thin_slice_quadrupole(test_context):

    quad = xt.Quadrupole(k1=0.1, length=1)

    line = xt.Line(elements=[quad])

    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Teapot(10000))])
    line.build_tracker(_context=test_context)
    assert line['e0..995'].parent_name == 'e0'
    assert line['e0..995']._parent is line.element_dict['e0']
    assert line['drift_e0..995'].parent_name == 'e0'
    assert line['drift_e0..995']._parent is line.element_dict['e0']

    p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03,
                        _context=test_context)
    p_ref = p0.copy()
    p_slice = p0.copy()

    line.track(p_slice)
    quad.track(p_ref)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-10)

    line.to_json('ttt_thin_quad.json')
    line2 = xt.load('ttt_thin_quad.json')
    assert isinstance(line2['e0..995'], xt.ThinSliceQuadrupole)
    assert line2['e0..995'].parent_name == 'e0'
    assert line2['e0..995']._parent is None
    assert line2['drift_e0..995'].parent_name == 'e0'
    assert line2['drift_e0..995']._parent is None

    line2.build_tracker(_context=test_context)
    assert isinstance(line2['e0..995'], xt.ThinSliceQuadrupole)
    assert line2['e0..995'].parent_name == 'e0'
    assert line2['e0..995']._parent is line2.element_dict['e0']
    assert isinstance(line2['drift_e0..995'], xt.DriftSliceQuadrupole)
    assert line2['drift_e0..995'].parent_name == 'e0'
    assert line2['drift_e0..995']._parent is line2.element_dict['e0']

    line.track(p_slice, backtrack=True)

    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-10)

    line.optimize_for_tracking()

    assert isinstance(line['e0..995'], xt.SimpleThinQuadrupole)
    assert isinstance(line['drift_e0..995'], xt.Drift)

    p_slice = p0.copy()
    line.track(p_slice)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-10)

    line.track(p_slice, backtrack=True)

    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-10)

@for_all_test_contexts
def test_thin_slice_sextupole(test_context):

    sext = xt.Sextupole(k2=0.1, length=1,
                shift_x=1e-3, shift_y=2e-3, rot_s_rad=0.2
                )

    line = xt.Line(elements=[sext])

    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Teapot(1))])
    line.build_tracker(_context=test_context)
    assert line['e0..0'].parent_name == 'e0'
    assert line['e0..0']._parent is line.element_dict['e0']
    assert line['drift_e0..0'].parent_name == 'e0'
    assert line['drift_e0..0']._parent is line.element_dict['e0']

    p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03,
                        _context=test_context)
    p_ref = p0.copy()
    p_slice = p0.copy()

    line.track(p_slice)
    sext.track(p_ref)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-10)

    line.to_json('ttt_thin_sext.json')
    line2 = xt.load('ttt_thin_sext.json')
    assert isinstance(line2['e0..0'], xt.ThinSliceSextupole)
    assert line2['e0..0'].parent_name == 'e0'
    assert line2['e0..0']._parent is None
    assert line2['drift_e0..0'].parent_name == 'e0'
    assert line2['drift_e0..0']._parent is None

    line2.build_tracker(_context=test_context)
    assert isinstance(line2['e0..0'], xt.ThinSliceSextupole)
    assert line2['e0..0'].parent_name == 'e0'
    assert line2['e0..0']._parent is line2.element_dict['e0']
    assert isinstance(line2['drift_e0..0'], xt.DriftSliceSextupole)
    assert line2['drift_e0..0'].parent_name == 'e0'
    assert line2['drift_e0..0']._parent is line2.element_dict['e0']

    line.track(p_slice, backtrack=True)

    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-10)

    line.optimize_for_tracking()

    assert isinstance(line['e0..0'], xt.Multipole)
    assert isinstance(line['drift_e0..0'], xt.Drift)

    p_slice = p0.copy()
    line.track(p_slice)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-10)

    line.track(p_slice, backtrack=True)

    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-10)

@for_all_test_contexts
def test_thin_slice_octupole(test_context):

    oct = xt.Octupole(k3=0.1, length=1)

    line = xt.Line(elements=[oct])

    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Teapot(1))])
    line.build_tracker(_context=test_context)
    assert line['e0..0'].parent_name == 'e0'
    assert line['e0..0']._parent is line.element_dict['e0']
    assert line['drift_e0..0'].parent_name == 'e0'
    assert line['drift_e0..0']._parent is line.element_dict['e0']

    p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03,
                        _context=test_context)
    p_ref = p0.copy()
    p_slice = p0.copy()

    line.track(p_slice)
    oct.track(p_ref)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-10)

    line.to_json('ttt_thin_oct.json')
    line2 = xt.load('ttt_thin_oct.json')
    assert isinstance(line2['e0..0'], xt.ThinSliceOctupole)
    assert line2['e0..0'].parent_name == 'e0'
    assert line2['e0..0']._parent is None
    assert line2['drift_e0..0'].parent_name == 'e0'
    assert line2['drift_e0..0']._parent is None

    line2.build_tracker(_context=test_context)
    assert isinstance(line2['e0..0'], xt.ThinSliceOctupole)
    assert line2['e0..0'].parent_name == 'e0'
    assert line2['e0..0']._parent is line2.element_dict['e0']
    assert isinstance(line2['drift_e0..0'], xt.DriftSliceOctupole)
    assert line2['drift_e0..0'].parent_name == 'e0'
    assert line2['drift_e0..0']._parent is line2.element_dict['e0']

    line.track(p_slice, backtrack=True)

    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-10)

    line.optimize_for_tracking()

    assert isinstance(line['e0..0'], xt.Multipole)
    assert isinstance(line['drift_e0..0'], xt.Drift)

    p_slice = p0.copy()
    line.track(p_slice)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-10)

    line.track(p_slice, backtrack=True)

    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-10)

@for_all_test_contexts
def test_thin_slice_drift(test_context):

    drift = xt.Drift(length=1)

    line = xt.Line(elements=[drift])

    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Uniform(5), element_type=xt.Drift)])
    line.build_tracker(_context=test_context)
    assert line['drift_e0..0'].parent_name == 'e0'
    assert line['drift_e0..0']._parent is line.element_dict['e0']

    p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03,
                      _context=test_context)
    p_ref = p0.copy()
    p_slice = p0.copy()

    line.track(p_slice)
    drift.track(p_ref)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-10)

    line.to_json('ttt_drift.json')
    line2 = xt.load('ttt_drift.json')
    assert line2['drift_e0..0'].parent_name == 'e0'
    assert line2['drift_e0..0']._parent is None

    line2.build_tracker(_context=test_context)
    assert isinstance(line2['drift_e0..0'], xt.DriftSlice)
    assert line2['drift_e0..0'].parent_name == 'e0'
    assert line2['drift_e0..0']._parent is line2.element_dict['e0']

    line.track(p_slice, backtrack=True)

    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-10)

    line.optimize_for_tracking()

    assert isinstance(line['drift_e0..0'], xt.Drift)

    p_slice = p0.copy()
    line.track(p_slice)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-10)

    line.track(p_slice, backtrack=True)

    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-10)


@pytest.mark.parametrize('bend_type', [xt.Bend, xt.RBend])
@for_all_test_contexts
def test_thick_slice_bend(test_context, bend_type):

    bend = bend_type(k0=0.4, h=0.3, length=1,
                     edge_entry_angle=0.05, edge_entry_hgap=0.06, edge_entry_fint=0.08,
                     edge_exit_angle=0.05, edge_exit_hgap=0.06, edge_exit_fint=0.08)

    line = xt.Line(elements=[bend])

    line.configure_bend_model(edge='linear', core='expanded')

    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Teapot(5, mode='thick'))])
    line.build_tracker(_context=test_context)
    line._line_before_slicing.build_tracker(_context=test_context)
    assert line['e0..3'].parent_name == 'e0'
    assert line['e0..3']._parent is line.element_dict['e0']
    assert line['e0..entry_map'].parent_name == 'e0'
    assert line['e0..entry_map']._parent is line.element_dict['e0']
    assert line['e0..exit_map'].parent_name == 'e0'
    assert line['e0..exit_map']._parent is line.element_dict['e0']

    p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03,
                        _context=test_context)
    p_ref = p0.copy()
    p_slice = p0.copy()

    line.track(p_slice)
    line._line_before_slicing.track(p_ref)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-10)

    thick_slice_cls, thick_slice_bend_cls = {
        xt.Bend: (
            xt.ThickSliceBend,
            xt.ThickSliceBend
        ),
        xt.RBend: (
            xt.ThickSliceRBend,
            xt.ThickSliceRBend
        ),
    }[bend_type]

    line.to_json('ttt_thick_bend.json')
    line2 = xt.load('ttt_thick_bend.json')
    assert isinstance(line2['e0..3'], thick_slice_cls)
    assert line2['e0..3'].parent_name == 'e0'
    assert line2['e0..3']._parent is None
    assert line2['e0..entry_map'].parent_name == 'e0'
    assert line2['e0..entry_map']._parent is None
    assert line2['e0..exit_map'].parent_name == 'e0'
    assert line2['e0..exit_map']._parent is None

    line2.build_tracker(_context=test_context)
    assert isinstance(line2['e0..3'], thick_slice_bend_cls)
    assert line2['e0..3'].parent_name == 'e0'
    assert line2['e0..3']._parent is line2.element_dict['e0']
    assert line2['e0..entry_map'].parent_name == 'e0'
    assert line2['e0..entry_map']._parent is line2.element_dict['e0']
    assert line2['e0..exit_map'].parent_name == 'e0'
    assert line2['e0..exit_map']._parent is line2.element_dict['e0']

    line.track(p_slice, backtrack=True)

    assert (p_slice.state == 1).all()
    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-10)

@for_all_test_contexts
def test_thick_slice_quadrupole(test_context):

    quad = xt.Quadrupole(k1=0.1, length=1)

    line = xt.Line(elements=[quad])

    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Teapot(5, mode='thick'))])
    line.build_tracker(_context=test_context)
    assert line['e0..3'].parent_name == 'e0'
    assert line['e0..3']._parent is line.element_dict['e0']

    p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03,
                        _context=test_context)
    p_ref = p0.copy()
    p_slice = p0.copy()

    line.track(p_slice)
    quad.track(p_ref)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-10)

    line.to_json('ttt_thick_quad.json')
    line2 = xt.load('ttt_thick_quad.json')
    assert isinstance(line2['e0..3'], xt.ThickSliceQuadrupole)
    assert line2['e0..3'].parent_name == 'e0'
    assert line2['e0..3']._parent is None

    line2.build_tracker(_context=test_context)
    assert isinstance(line2['e0..3'], xt.ThickSliceQuadrupole)
    assert line2['e0..3'].parent_name == 'e0'
    assert line2['e0..3']._parent is line2.element_dict['e0']

    line.track(p_slice, backtrack=True)

    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-10)

@for_all_test_contexts
def test_thick_slice_sextupole(test_context):

    sext = xt.Sextupole(k2=0.1, length=1)

    line = xt.Line(elements=[sext])

    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Teapot(1, mode='thick'))])
    line.build_tracker(_context=test_context)
    assert line['e0..0'].parent_name == 'e0'
    assert line['e0..0']._parent is line.element_dict['e0']

    p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03,
                        _context=test_context)
    p_ref = p0.copy()
    p_slice = p0.copy()

    line.track(p_slice)
    sext.track(p_ref)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-10)

    line.to_json('ttt_thick_sext.json')
    line2 = xt.load('ttt_thick_sext.json')
    assert isinstance(line2['e0..0'], xt.ThickSliceSextupole)
    assert line2['e0..0'].parent_name == 'e0'
    assert line2['e0..0']._parent is None

    line2.build_tracker(_context=test_context)
    assert isinstance(line2['e0..0'], xt.ThickSliceSextupole)
    assert line2['e0..0'].parent_name == 'e0'
    assert line2['e0..0']._parent is line2.element_dict['e0']

    line.track(p_slice, backtrack=True)

    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-10)

@for_all_test_contexts
def test_thick_slice_octupole(test_context):

    oct = xt.Octupole(k3=0.1, length=1)

    line = xt.Line(elements=[oct])

    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Teapot(1, mode='thick'))])
    line.build_tracker(_context=test_context)
    assert line['e0..0'].parent_name == 'e0'
    assert line['e0..0']._parent is line.element_dict['e0']

    p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03,
                        _context=test_context)
    p_ref = p0.copy()
    p_slice = p0.copy()

    line.track(p_slice)
    oct.track(p_ref)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-10)

    line.to_json('ttt_thick_oct.json')
    line2 = xt.load('ttt_thick_oct.json')
    assert isinstance(line2['e0..0'], xt.ThickSliceOctupole)
    assert line2['e0..0'].parent_name == 'e0'
    assert line2['e0..0']._parent is None

    line2.build_tracker(_context=test_context)
    assert isinstance(line2['e0..0'], xt.ThickSliceOctupole)
    assert line2['e0..0'].parent_name == 'e0'
    assert line2['e0..0']._parent is line2.element_dict['e0']

    line.track(p_slice, backtrack=True)

    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-10)

@for_all_test_contexts
def test_thick_slice_solenoid(test_context):

    sol = xt.UniformSolenoid(ks=0.1, length=1)

    line = xt.Line(elements=[sol])

    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Teapot(10, mode='thick'))])
    line.build_tracker(_context=test_context)
    assert line['e0..0'].parent_name == 'e0'
    assert line['e0..0']._parent is line.element_dict['e0']

    p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03,
                        _context=test_context)
    p_ref = p0.copy()
    p_slice = p0.copy()

    line.track(p_slice)
    sol.track(p_ref)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-10)

    line.to_json('ttt_thick_solenoid.json')
    line2 = xt.load('ttt_thick_solenoid.json')
    assert isinstance(line2['e0..0'], xt.ThickSliceUniformSolenoid)
    assert line2['e0..0'].parent_name == 'e0'
    assert line2['e0..0']._parent is None

    line2.build_tracker(_context=test_context)
    assert isinstance(line2['e0..0'], xt.ThickSliceUniformSolenoid)
    assert line2['e0..0'].parent_name == 'e0'
    assert line2['e0..0']._parent is line2.element_dict['e0']

    line.track(p_slice, backtrack=True)

    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-10)


@for_all_test_contexts
@pytest.mark.parametrize(
    'model',
    ['full', 'linear'])
def test_bend_edge_slice_entry(test_context, model):

    # only edge entry
    bend_only_e1 = xt.Bend(
        length=0, k0=0.1,
        edge_entry_angle=0.05,
        edge_entry_hgap=0.06,
        edge_entry_fint=0.08,
        edge_exit_active=False,
        _context=test_context)

    edge_e1 = xt.DipoleEdge(
        k=0.1, side='entry', e1=0.05, hgap=0.06, fint=0.08,
        _context=test_context)

    line = xt.Line(elements=[bend_only_e1])

    line['e0'].length = 1 # to force the slicing
    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Teapot(1))])
    line['e0'].length = 0
    line.build_tracker(_context=test_context)

    assert 'e0..entry_map' in line.element_names
    assert 'e0..exit_map' in line.element_names
    assert isinstance(line['e0..entry_map'], xt.ThinSliceBendEntry)
    assert isinstance(line['e0..exit_map'], xt.ThinSliceBendExit)

    edge_e1.model = model
    bend_only_e1.edge_entry_model = model

    p1 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03,
                      _context=test_context)
    p2 = p1.copy()
    p3 = p1.copy()

    bend_only_e1.track(p1)
    edge_e1.track(p2)
    line.track(p3)

    assert_allclose(p1.x, p2.x, rtol=0, atol=1e-14)
    assert_allclose(p1.px, p2.px, rtol=0, atol=1e-14)
    assert_allclose(p1.y, p2.y, rtol=0, atol=1e-14)
    assert_allclose(p1.py, p2.py, rtol=0, atol=1e-14)
    assert_allclose(p1.zeta, p2.zeta, rtol=0, atol=1e-14)
    assert_allclose(p1.delta, p2.delta, rtol=0, atol=1e-14)

    assert_allclose(p1.x, p3.x, rtol=0, atol=1e-14)
    assert_allclose(p1.px, p3.px, rtol=0, atol=1e-14)
    assert_allclose(p1.y, p3.y, rtol=0, atol=1e-14)
    assert_allclose(p1.py, p3.py, rtol=0, atol=1e-14)
    assert_allclose(p1.zeta, p3.zeta, rtol=0, atol=1e-14)
    assert_allclose(p1.delta, p3.delta, rtol=0, atol=1e-14)

@for_all_test_contexts
@pytest.mark.parametrize(
    'model',
    ['full', 'linear'])
def test_bend_edge_slice_exit(test_context, model):

    # only edge entry
    bend_only_e1 = xt.Bend(
        length=0, k0=0.1,
        edge_entry_angle=0.05,
        edge_entry_hgap=0.06,
        edge_entry_fint=0.08,
        edge_exit_active=False,
        _context=test_context)

    edge_e1 = xt.DipoleEdge(
        k=0.1, side='entry', e1=0.05, hgap=0.06, fint=0.08, _context=test_context)

    line = xt.Line(elements=[bend_only_e1])

    line['e0'].length = 1 # to force the slicing
    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Teapot(1))])
    line['e0'].length = 0
    line.build_tracker(_context=test_context)

    assert 'e0..entry_map' in line.element_names
    assert 'e0..exit_map' in line.element_names
    assert isinstance(line['e0..entry_map'], xt.ThinSliceBendEntry)
    assert isinstance(line['e0..exit_map'], xt.ThinSliceBendExit)

    edge_e1.model = model
    bend_only_e1.edge_entry_model = model

    p1 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03,
                      _context=test_context)
    p2 = p1.copy()
    p3 = p1.copy()

    bend_only_e1.track(p1)
    edge_e1.track(p2)
    line.track(p3)

    assert_allclose(p1.x, p2.x, rtol=0, atol=1e-14)
    assert_allclose(p1.px, p2.px, rtol=0, atol=1e-14)
    assert_allclose(p1.y, p2.y, rtol=0, atol=1e-14)
    assert_allclose(p1.py, p2.py, rtol=0, atol=1e-14)
    assert_allclose(p1.zeta, p2.zeta, rtol=0, atol=1e-14)
    assert_allclose(p1.delta, p2.delta, rtol=0, atol=1e-14)

    assert_allclose(p1.x, p3.x, rtol=0, atol=1e-14)
    assert_allclose(p1.px, p3.px, rtol=0, atol=1e-14)
    assert_allclose(p1.y, p3.y, rtol=0, atol=1e-14)
    assert_allclose(p1.py, p3.py, rtol=0, atol=1e-14)
    assert_allclose(p1.zeta, p3.zeta, rtol=0, atol=1e-14)
    assert_allclose(p1.delta, p3.delta, rtol=0, atol=1e-14)

@for_all_test_contexts
def test_thin_slice_bend_with_multipoles(test_context):

    bend = xt.Bend(k0=0.4, h=0.3, length=1,
                   k1=0.003,
                   knl=[0, 0.001, 0.01, 0.02, 0.04, 0.6],
                   ksl=[0, 0.002, 0.03, 0.03, 0.05, 0.7],
                   num_multipole_kicks=1000,
                   edge_entry_angle=0.05, edge_entry_hgap=0.06, edge_entry_fint=0.08,
                   edge_exit_angle=0.05, edge_exit_hgap=0.06, edge_exit_fint=0.08)

    line = xt.Line(elements=[bend])
    bend.integrator = 'teapot'
    bend.model = 'drift-kick-drift-expanded'

    line.configure_bend_model(edge='linear')

    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Teapot(1000))])
    line.build_tracker(_context=test_context)
    line._line_before_slicing.build_tracker(_context=test_context)
    assert line['e0..995'].parent_name == 'e0'
    assert line['e0..995']._parent is line.element_dict['e0']
    assert line['drift_e0..995'].parent_name == 'e0'
    assert line['drift_e0..995']._parent is line.element_dict['e0']
    assert line['e0..entry_map'].parent_name == 'e0'
    assert line['e0..entry_map']._parent is line.element_dict['e0']
    assert line['e0..exit_map'].parent_name == 'e0'
    assert line['e0..exit_map']._parent is line.element_dict['e0']

    p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03
                      ,_context=test_context)
    p_ref = p0.copy()
    p_slice = p0.copy()

    line.track(p_slice)
    line._line_before_slicing.track(p_ref)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-8)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-8)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-8)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-8)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-8)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-8)

    line.to_json('ttt_bend_mult.json')
    line2 = xt.load('ttt_bend_mult.json')
    assert isinstance(line2['e0..995'], xt.ThinSliceBend)
    assert line2['e0..995'].parent_name == 'e0'
    assert line2['e0..995']._parent is None
    assert line2['drift_e0..995'].parent_name == 'e0'
    assert line2['drift_e0..995']._parent is None
    assert line2['e0..entry_map'].parent_name == 'e0'
    assert line2['e0..entry_map']._parent is None
    assert line2['e0..exit_map'].parent_name == 'e0'
    assert line2['e0..exit_map']._parent is None

    line2.build_tracker(_context=test_context)
    assert isinstance(line2['e0..995'], xt.ThinSliceBend)
    assert line2['e0..995'].parent_name == 'e0'
    assert line2['e0..995']._parent is line2.element_dict['e0']
    assert isinstance(line2['drift_e0..995'], xt.DriftSliceBend)
    assert line2['drift_e0..995'].parent_name == 'e0'
    assert line2['drift_e0..995']._parent is line2.element_dict['e0']
    assert isinstance(line2['e0..entry_map'], xt.ThinSliceBendEntry)
    assert line2['e0..entry_map'].parent_name == 'e0'
    assert line2['e0..entry_map']._parent is line2.element_dict['e0']
    assert isinstance(line2['e0..exit_map'], xt.ThinSliceBendExit)
    assert line2['e0..exit_map'].parent_name == 'e0'
    assert line2['e0..exit_map']._parent is line2.element_dict['e0']

    line.track(p_slice, backtrack=True)

    assert (p_slice.state == 1).all()
    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-10)

    line.optimize_for_tracking()

    assert isinstance(line['e0..995'], xt.Multipole)
    assert isinstance(line['drift_e0..995'], xt.Drift)

    assert isinstance(line['e0..entry_map'], xt.DipoleEdge)
    assert isinstance(line['e0..exit_map'], xt.DipoleEdge)

    p_slice = p0.copy()
    line.track(p_slice)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-8)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-8)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-8)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-8)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-8)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-8)

    line.track(p_slice, backtrack=True)

    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-10)

@for_all_test_contexts
def test_thick_slice_bend_with_multipoles(test_context):

    bend = xt.Bend(k0=0.4, h=0.3, length=1,
                   k1=0.003,
                   knl=[0, 0.001, 0.01, 0.02, 0.04, 0.6],
                   ksl=[0, 0.002, 0.03, 0.03, 0.05, 0.7],
                   num_multipole_kicks=100000,
                   edge_entry_angle=0.05, edge_entry_hgap=0.06, edge_entry_fint=0.08,
                   edge_exit_angle=0.05, edge_exit_hgap=0.06, edge_exit_fint=0.08)

    line = xt.Line(elements=[bend])

    line.configure_bend_model(edge='linear', core='expanded')

    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Teapot(10000, mode='thick'))])
    line.build_tracker(_context=test_context)
    line._line_before_slicing.build_tracker(_context=test_context)
    assert line['e0..995'].parent_name == 'e0'
    assert line['e0..995']._parent is line.element_dict['e0']
    assert line['e0..entry_map'].parent_name == 'e0'
    assert line['e0..entry_map']._parent is line.element_dict['e0']
    assert line['e0..exit_map'].parent_name == 'e0'
    assert line['e0..exit_map']._parent is line.element_dict['e0']

    p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03
                      ,_context=test_context)
    p_ref = p0.copy()
    p_slice = p0.copy()

    line.track(p_slice)
    line._line_before_slicing.track(p_ref)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-8)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-8)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-8)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-8)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-8)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-8)

    line.to_json('ttt_thick_bend_mult.json')
    line2 = xt.load('ttt_thick_bend_mult.json')
    assert isinstance(line2['e0..995'], xt.ThickSliceBend)
    assert line2['e0..995'].parent_name == 'e0'
    assert line2['e0..995']._parent is None
    assert line2['e0..entry_map'].parent_name == 'e0'
    assert line2['e0..entry_map']._parent is None
    assert line2['e0..exit_map'].parent_name == 'e0'
    assert line2['e0..exit_map']._parent is None

    line2.build_tracker(_context=test_context)
    assert isinstance(line2['e0..995'], xt.ThickSliceBend)
    assert line2['e0..995'].parent_name == 'e0'
    assert line2['e0..995']._parent is line2.element_dict['e0']
    assert isinstance(line2['e0..entry_map'], xt.ThinSliceBendEntry)
    assert line2['e0..entry_map'].parent_name == 'e0'
    assert line2['e0..entry_map']._parent is line2.element_dict['e0']
    assert isinstance(line2['e0..exit_map'], xt.ThinSliceBendExit)
    assert line2['e0..exit_map'].parent_name == 'e0'
    assert line2['e0..exit_map']._parent is line2.element_dict['e0']

    line.track(p_slice, backtrack=True)

    assert (p_slice.state == 1).all()
    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-10)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-10)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-10)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-10)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-10)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-10)

@for_all_test_contexts
def test_thin_slice_bend_with_multipoles_bend_off(test_context):

    num_slices = 10

    bend = xt.Bend(k0=0, h=0, length=1,
                   knl=[0, 0.001, 0.01, 0.02, 0.04, 0.6],
                   ksl=[0, 0.002, 0.03, 0.03, 0.05, 0.7],
                   num_multipole_kicks=num_slices,
                   edge_entry_angle=0.05, edge_entry_hgap=0.06, edge_entry_fint=0.08,
                   edge_exit_angle=0.05, edge_exit_hgap=0.06, edge_exit_fint=0.08)
    bend.integrator = 'teapot'

    line = xt.Line(elements=[bend])

    line.configure_bend_model(edge='linear', core='expanded')


    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Teapot(num_slices))])
    line.build_tracker(_context=test_context)
    line._line_before_slicing.build_tracker(_context=test_context)
    assert line['e0..5'].parent_name == 'e0'
    assert line['e0..5']._parent is line.element_dict['e0']
    assert line['drift_e0..5'].parent_name == 'e0'
    assert line['drift_e0..5']._parent is line.element_dict['e0']
    assert line['e0..entry_map'].parent_name == 'e0'
    assert line['e0..entry_map']._parent is line.element_dict['e0']
    assert line['e0..exit_map'].parent_name == 'e0'
    assert line['e0..exit_map']._parent is line.element_dict['e0']

    p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03
                      ,_context=test_context)
    p_ref = p0.copy()
    p_slice = p0.copy()

    line.track(p_slice)
    line._line_before_slicing.track(p_ref)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-14)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-14)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-14)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-14)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-14)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-14)

    line.to_json('ttt_thin_bend_mult_off.json')
    line2 = xt.load('ttt_thin_bend_mult_off.json')
    assert isinstance(line2['e0..5'], xt.ThinSliceBend)
    assert line2['e0..5'].parent_name == 'e0'
    assert line2['e0..5']._parent is None
    assert line2['drift_e0..5'].parent_name == 'e0'
    assert line2['drift_e0..5']._parent is None
    assert line2['e0..entry_map'].parent_name == 'e0'
    assert line2['e0..entry_map']._parent is None
    assert line2['e0..exit_map'].parent_name == 'e0'
    assert line2['e0..exit_map']._parent is None

    line2.build_tracker(_context=test_context)
    assert isinstance(line2['e0..5'], xt.ThinSliceBend)
    assert line2['e0..5'].parent_name == 'e0'
    assert line2['e0..5']._parent is line2.element_dict['e0']
    assert isinstance(line2['drift_e0..5'], xt.DriftSliceBend)
    assert line2['drift_e0..5'].parent_name == 'e0'
    assert line2['drift_e0..5']._parent is line2.element_dict['e0']
    assert isinstance(line2['e0..entry_map'], xt.ThinSliceBendEntry)
    assert line2['e0..entry_map'].parent_name == 'e0'
    assert line2['e0..entry_map']._parent is line2.element_dict['e0']
    assert isinstance(line2['e0..exit_map'], xt.ThinSliceBendExit)
    assert line2['e0..exit_map'].parent_name == 'e0'
    assert line2['e0..exit_map']._parent is line2.element_dict['e0']

    line.track(p_slice, backtrack=True)

    assert (p_slice.state == 1).all()
    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-14)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-14)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-14)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-14)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-14)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-14)

    line.optimize_for_tracking()

    assert_allclose(line['e0..5'].knl[5], 0.6/num_slices, rtol=0, atol=1e-14)
    assert_allclose(line['e0..5'].ksl[5], 0.7/num_slices, rtol=0, atol=1e-14)

    assert isinstance(line['e0..5'], xt.Multipole)
    assert isinstance(line['drift_e0..5'], xt.Drift)

    assert isinstance(line['e0..entry_map'], xt.DipoleEdge)
    assert isinstance(line['e0..exit_map'], xt.DipoleEdge)

    p_slice = p0.copy()
    line.track(p_slice)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-14)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-14)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-14)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-14)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-14)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-14)

    line.track(p_slice, backtrack=True)

    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-14)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-14)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-14)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-14)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-14)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-14)

@for_all_test_contexts
def test_thick_slice_quad_with_multipoles(test_context):

    quad = xt.Quadrupole(k1=1e-3, k1s=2e-3, length=1,
                   knl=[0, 0.001, 0.01, 0.02, 0.04, 0.6],
                   ksl=[0, 0.002, 0.03, 0.03, 0.05, 0.7],
                   num_multipole_kicks=1000)

    line = xt.Line(elements=[quad])

    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Uniform(1000, mode='thick'))])
    line.build_tracker(_context=test_context)
    line._line_before_slicing.build_tracker(_context=test_context)
    assert line['e0..5'].parent_name == 'e0'
    assert line['e0..5']._parent is line.element_dict['e0']

    p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03
                      ,_context=test_context)
    p_ref = p0.copy()
    p_slice = p0.copy()

    line.track(p_slice)
    line._line_before_slicing.track(p_ref)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-8)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-8)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-8)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-8)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-8)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-8)

    line.to_json('ttt_thick_quad_mult.json')
    line2 = xt.load('ttt_thick_quad_mult.json')
    assert isinstance(line2['e0..5'], xt.ThickSliceQuadrupole)
    assert line2['e0..5'].parent_name == 'e0'
    assert line2['e0..5']._parent is None

    line2.build_tracker(_context=test_context)
    assert isinstance(line2['e0..5'], xt.ThickSliceQuadrupole)
    assert line2['e0..5'].parent_name == 'e0'
    assert line2['e0..5']._parent is line2.element_dict['e0']

    line.track(p_slice, backtrack=True)

    assert (p_slice.state == 1).all()
    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-8)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-8)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-8)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-8)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-8)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-8)

@for_all_test_contexts
def test_thin_slice_quad_with_multipoles(test_context):

    quad = xt.Quadrupole(k1=1e-3, k1s=2e-3, length=1,
                   knl=[0, 0.001, 0.01, 0.02, 0.04, 0.6],
                   ksl=[0, 0.002, 0.03, 0.03, 0.05, 0.7],
                   num_multipole_kicks=7)
    quad.integrator = 'teapot'
    quad.model = 'drift-kick-drift-expanded'

    line = xt.Line(elements=[quad])

    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Teapot(7))])
    line.build_tracker(_context=test_context)
    line._line_before_slicing.build_tracker(_context=test_context)
    assert line['e0..5'].parent_name == 'e0'
    assert line['e0..5']._parent is line.element_dict['e0']
    assert line['drift_e0..5'].parent_name == 'e0'
    assert line['drift_e0..5']._parent is line.element_dict['e0']

    p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03
                      ,_context=test_context)
    p_ref = p0.copy()
    p_slice = p0.copy()

    line.track(p_slice)
    line._line_before_slicing.track(p_ref)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-8)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-8)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-8)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-8)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-8)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-8)

    line.to_json('ttt_thin_quad_mult.json')
    line2 = xt.load('ttt_thin_quad_mult.json')
    assert isinstance(line2['e0..5'], xt.ThinSliceQuadrupole)
    assert line2['e0..5'].parent_name == 'e0'
    assert line2['e0..5']._parent is None
    assert line2['drift_e0..5'].parent_name == 'e0'
    assert line2['drift_e0..5']._parent is None

    line2.build_tracker(_context=test_context)
    assert isinstance(line2['e0..5'], xt.ThinSliceQuadrupole)
    assert line2['e0..5'].parent_name == 'e0'
    assert line2['e0..5']._parent is line2.element_dict['e0']
    assert isinstance(line2['drift_e0..5'], xt.DriftSliceQuadrupole)
    assert line2['drift_e0..5'].parent_name == 'e0'
    assert line2['drift_e0..5']._parent is line2.element_dict['e0']

    line.track(p_slice, backtrack=True)

    assert (p_slice.state == 1).all()
    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-8)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-8)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-8)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-8)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-8)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-8)

    line.optimize_for_tracking()

    assert isinstance(line['e0..5'], xt.Multipole)
    assert isinstance(line['drift_e0..5'], xt.Drift)

    p_slice = p0.copy()
    line.track(p_slice)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-8)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-8)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-8)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-8)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-8)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-8)

    line.track(p_slice, backtrack=True)

    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-14)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-14)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-14)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-14)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-14)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-14)

@for_all_test_contexts
def test_thin_slice_quad_with_multipoles_quad_off(test_context):

    quad = xt.Quadrupole(k1=0, k1s=0, length=1,
                   knl=[0, 0.001, 0.01, 0.02, 0.04, 0.6],
                   ksl=[0, 0.002, 0.03, 0.03, 0.05, 0.7],
                   num_multipole_kicks=7)
    quad.model = 'drift-kick-drift-expanded'
    quad.integrator = 'teapot'

    line = xt.Line(elements=[quad])

    num_slices = 7

    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Teapot(num_slices))])
    line.build_tracker(_context=test_context)
    line._line_before_slicing.build_tracker(_context=test_context)
    assert line['e0..5'].parent_name == 'e0'
    assert line['e0..5']._parent is line.element_dict['e0']
    assert line['drift_e0..5'].parent_name == 'e0'
    assert line['drift_e0..5']._parent is line.element_dict['e0']

    p0 = xt.Particles(p0c=10e9, x=0.1, px=0.5, y=0.3, py=0.3, delta=0.1
                      ,_context=test_context)
    p_ref = p0.copy()
    p_slice = p0.copy()

    line.track(p_slice)
    line._line_before_slicing.track(p_ref)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-14)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-14)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-14)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-14)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-14)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-14)

    line.to_json('ttt_thin_mult_quad_off.json')
    line2 = xt.load('ttt_thin_mult_quad_off.json')
    assert isinstance(line2['e0..5'], xt.ThinSliceQuadrupole)
    assert line2['e0..5'].parent_name == 'e0'
    assert line2['e0..5']._parent is None
    assert line2['drift_e0..5'].parent_name == 'e0'
    assert line2['drift_e0..5']._parent is None

    line2.build_tracker(_context=test_context)
    assert isinstance(line2['e0..5'], xt.ThinSliceQuadrupole)
    assert line2['e0..5'].parent_name == 'e0'
    assert line2['e0..5']._parent is line2.element_dict['e0']
    assert isinstance(line2['drift_e0..5'], xt.DriftSliceQuadrupole)
    assert line2['drift_e0..5'].parent_name == 'e0'
    assert line2['drift_e0..5']._parent is line2.element_dict['e0']

    line.track(p_slice, backtrack=True)

    assert (p_slice.state == 1).all()
    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-14)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-14)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-14)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-14)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-14)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-14)

    line.optimize_for_tracking()

    assert isinstance(line['e0..5'], xt.Multipole)
    assert isinstance(line['drift_e0..5'], xt.Drift)

    assert_allclose(line['e0..5'].knl[5], 0.6/num_slices, rtol=0, atol=1e-14)
    assert_allclose(line['e0..5'].ksl[5], 0.7/num_slices, rtol=0, atol=1e-14)

    p_slice = p0.copy()
    line.track(p_slice)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-14)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-14)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-14)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-14)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-14)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-14)

    line.track(p_slice, backtrack=True)

    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-14)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-14)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-14)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-14)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-14)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-14)

@for_all_test_contexts
def test_thick_slice_sextupole_with_multipoles(test_context):

    sext = xt.Sextupole(k2=1e-3, k2s=2e-3,
                   length=0.001, # need to make it very short because thick has only one kick in the center
                   knl=[0, 0.001, 0.01, 0.02, 0.04, 0.6],
                   ksl=[0, 0.002, 0.03, 0.03, 0.05, 0.7]
    )

    line = xt.Line(elements=[sext])

    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Uniform(2, mode='thick'))])
    line.build_tracker(_context=test_context)
    line._line_before_slicing.build_tracker(_context=test_context)
    assert line['e0..1'].parent_name == 'e0'
    assert line['e0..1']._parent is line.element_dict['e0']

    p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03
                      ,_context=test_context)
    p_ref = p0.copy()
    p_slice = p0.copy()

    line.track(p_slice)
    line._line_before_slicing.track(p_ref)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-8)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-8)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-8)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-8)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-8)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-8)

    line.to_json('ttt_thick_sext_mult.json')
    line2 = xt.load('ttt_thick_sext_mult.json')
    assert isinstance(line2['e0..0'], xt.ThickSliceSextupole)
    assert line2['e0..1'].parent_name == 'e0'
    assert line2['e0..1']._parent is None

    line2.build_tracker(_context=test_context)
    assert isinstance(line2['e0..0'], xt.ThickSliceSextupole)
    assert line2['e0..1'].parent_name == 'e0'
    assert line2['e0..1']._parent is line2.element_dict['e0']

    line.track(p_slice, backtrack=True)

    assert (p_slice.state == 1).all()
    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-8)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-8)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-8)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-8)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-8)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-8)

@for_all_test_contexts
def test_thin_slice_sextupole_with_multipoles(test_context):

    sext = xt.Sextupole(k2=1e-3, k2s=2e-3, length=0.001,
                   knl=[0, 0.001, 0.01, 0.02, 0.04, 0.6],
                   ksl=[0, 0.002, 0.03, 0.03, 0.05, 0.7])

    line = xt.Line(elements=[sext])

    num_slices = 2

    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Uniform(num_slices))])
    line.build_tracker(_context=test_context)
    line._line_before_slicing.build_tracker(_context=test_context)
    assert line['e0..1'].parent_name == 'e0'
    assert line['e0..1']._parent is line.element_dict['e0']
    assert line['drift_e0..1'].parent_name == 'e0'
    assert line['drift_e0..1']._parent is line.element_dict['e0']

    p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03
                      ,_context=test_context)
    p_ref = p0.copy()
    p_slice = p0.copy()

    line.track(p_slice)
    line._line_before_slicing.track(p_ref)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-8)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-8)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-8)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-8)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-8)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-8)

    line.to_json('ttt_thin_sext_mult.json')
    line2 = xt.load('ttt_thin_sext_mult.json')
    assert isinstance(line2['e0..1'], xt.ThinSliceSextupole)
    assert line2['e0..1'].parent_name == 'e0'
    assert line2['e0..1']._parent is None
    assert line2['drift_e0..1'].parent_name == 'e0'
    assert line2['drift_e0..1']._parent is None

    line2.build_tracker(_context=test_context)
    assert isinstance(line2['e0..1'], xt.ThinSliceSextupole)
    assert line2['e0..1'].parent_name == 'e0'
    assert line2['e0..1']._parent is line2.element_dict['e0']
    assert isinstance(line2['drift_e0..1'], xt.DriftSliceSextupole)
    assert line2['drift_e0..1'].parent_name == 'e0'
    assert line2['drift_e0..1']._parent is line2.element_dict['e0']

    line.track(p_slice, backtrack=True)

    assert (p_slice.state == 1).all()
    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-8)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-8)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-8)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-8)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-8)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-8)

    line.optimize_for_tracking()

    assert isinstance(line['e0..1'], xt.Multipole)
    assert isinstance(line['drift_e0..1'], xt.Drift)

    assert_allclose(line['e0..1'].knl[5], 0.6/num_slices, rtol=0, atol=1e-14)
    assert_allclose(line['e0..1'].ksl[5], 0.7/num_slices, rtol=0, atol=1e-14)

    p_slice = p0.copy()
    line.track(p_slice)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-8)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-8)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-8)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-8)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-8)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-8)

    line.track(p_slice, backtrack=True)

    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-14)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-14)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-14)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-14)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-14)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-14)

@for_all_test_contexts
def test_thick_slice_octupole_with_multipoles(test_context):

    oct = xt.Octupole(k3=1e-3, k3s=2e-3,
                   length=0.001, # need to make it very short because thick has only one kick in the center
                   knl=[0, 0.001, 0.01, 0.02, 0.04, 0.6],
                   ksl=[0, 0.002, 0.03, 0.03, 0.05, 0.7])

    line = xt.Line(elements=[oct])

    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Uniform(2, mode='thick'))])
    line.build_tracker(_context=test_context)
    line._line_before_slicing.build_tracker(_context=test_context)
    assert line['e0..1'].parent_name == 'e0'
    assert line['e0..1']._parent is line.element_dict['e0']

    p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03
                      ,_context=test_context)
    p_ref = p0.copy()
    p_slice = p0.copy()

    line.track(p_slice)
    line._line_before_slicing.track(p_ref)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-8)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-8)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-8)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-8)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-8)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-8)

    line.to_json('ttt_thick_oct_mult.json')
    line2 = xt.load('ttt_thick_oct_mult.json')
    assert isinstance(line2['e0..0'], xt.ThickSliceOctupole)
    assert line2['e0..1'].parent_name == 'e0'
    assert line2['e0..1']._parent is None

    line2.build_tracker(_context=test_context)
    assert isinstance(line2['e0..0'], xt.ThickSliceOctupole)
    assert line2['e0..1'].parent_name == 'e0'
    assert line2['e0..1']._parent is line2.element_dict['e0']

    line.track(p_slice, backtrack=True)

    assert (p_slice.state == 1).all()
    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-8)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-8)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-8)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-8)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-8)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-8)


@for_all_test_contexts
def test_thin_slice_octupole_with_multipoles(test_context):

    oct = xt.Octupole(k3=1e-3, k3s=-1e-3, length=0.001,
                   knl=[0, 0.001, 0.01, 0.02, 0.04, 0.6],
                   ksl=[0, 0.002, 0.03, 0.03, 0.05, 0.7])

    line = xt.Line(elements=[oct])

    num_slices = 2

    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Uniform(num_slices))])
    line.build_tracker(_context=test_context)
    line._line_before_slicing.build_tracker(_context=test_context)
    assert line['e0..1'].parent_name == 'e0'
    assert line['e0..1']._parent is line.element_dict['e0']
    assert line['drift_e0..1'].parent_name == 'e0'
    assert line['drift_e0..1']._parent is line.element_dict['e0']

    p0 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03
                      ,_context=test_context)
    p_ref = p0.copy()
    p_slice = p0.copy()

    line.track(p_slice)
    line._line_before_slicing.track(p_ref)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-8)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-8)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-8)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-8)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-8)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-8)

    line.to_json('ttt_octupole_thin.json')
    line2 = xt.load('ttt_octupole_thin.json')
    assert isinstance(line2['e0..1'], xt.ThinSliceOctupole)
    assert line2['e0..1'].parent_name == 'e0'
    assert line2['e0..1']._parent is None
    assert line2['drift_e0..1'].parent_name == 'e0'
    assert line2['drift_e0..1']._parent is None

    line2.build_tracker(_context=test_context)
    assert isinstance(line2['e0..1'], xt.ThinSliceOctupole)
    assert line2['e0..1'].parent_name == 'e0'
    assert line2['e0..1']._parent is line2.element_dict['e0']
    assert isinstance(line2['drift_e0..1'], xt.DriftSliceOctupole)
    assert line2['drift_e0..1'].parent_name == 'e0'
    assert line2['drift_e0..1']._parent is line2.element_dict['e0']

    line.track(p_slice, backtrack=True)

    assert (p_slice.state == 1).all()
    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-8)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-8)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-8)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-8)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-8)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-8)

    line.optimize_for_tracking()

    assert isinstance(line['e0..1'], xt.Multipole)
    assert isinstance(line['drift_e0..1'], xt.Drift)

    assert_allclose(line['e0..1'].knl[5], 0.6/num_slices, rtol=0, atol=1e-14)
    assert_allclose(line['e0..1'].ksl[5], 0.7/num_slices, rtol=0, atol=1e-14)

    p_slice = p0.copy()
    line.track(p_slice)

    assert_allclose(p_slice.x, p_ref.x, rtol=0, atol=1e-8)
    assert_allclose(p_slice.px, p_ref.px, rtol=0, atol=1e-8)
    assert_allclose(p_slice.y, p_ref.y, rtol=0, atol=1e-8)
    assert_allclose(p_slice.py, p_ref.py, rtol=0, atol=1e-8)
    assert_allclose(p_slice.zeta, p_ref.zeta, rtol=0, atol=1e-8)
    assert_allclose(p_slice.delta, p_ref.delta, rtol=0, atol=1e-8)

    line.track(p_slice, backtrack=True)

    assert_allclose(p_slice.x, p0.x, rtol=0, atol=1e-14)
    assert_allclose(p_slice.px, p0.px, rtol=0, atol=1e-14)
    assert_allclose(p_slice.y, p0.y, rtol=0, atol=1e-14)
    assert_allclose(p_slice.py, p0.py, rtol=0, atol=1e-14)
    assert_allclose(p_slice.zeta, p0.zeta, rtol=0, atol=1e-14)
    assert_allclose(p_slice.delta, p0.delta, rtol=0, atol=1e-14)


@pytest.mark.parametrize('element_class', ['quadrupole', 'octupole', 'sextupole'])
def test_sliced_magnet_fringes(element_class):
    env = xt.Environment()

    if element_class == 'quadrupole':
        element = env.new('qd', xt.Quadrupole, k1=0.1, length=0.1)
    elif element_class == 'sextupole':
        element = env.new('sd', xt.Sextupole, k2=0.1, length=0.1)
    elif element_class == 'octupole':
        element = env.new('od', xt.Octupole, k3=0.1, length=0.1)

    line = env.new_line(name='line', components=[element])
    line_thick = line.copy(shallow=True)
    line.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=xt.Teapot(3))])

    p0 = xt.Particles(kinetic_energy0=100e6, mass0=xt.PROTON_MASS_EV,
                      x=1e-3, px=1e-3, y=1e-3, py=1e-3, zeta=1e-3, delta=1e-3)

    p_ref_no_fringe = p0.copy()
    p_slice_no_fringe = p0.copy()

    line_thick.track(p_ref_no_fringe)
    line.track(p_slice_no_fringe)

    line.configure_quadrupole_model(edge='full')
    p_ref_fringe = p0.copy()
    p_slice_fringe = p0.copy()

    line_thick.track(p_ref_fringe)
    line.track(p_slice_fringe)

    diff_no_fringe_x = p_ref_no_fringe.x - p_slice_no_fringe.x
    diff_fringe_x = p_ref_fringe.x - p_slice_fringe.x

    xo.assert_allclose(diff_no_fringe_x, diff_fringe_x, rtol=0, atol=1e-16)

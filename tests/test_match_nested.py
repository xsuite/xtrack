import pathlib

import numpy as np

import xdeps as xd
import xobjects as xo
import xtrack
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

class ActionArcPhaseAdvanceFromCell(xd.Action):
    def __init__(self, arc_name, line_name, line):
        assert arc_name in ['12', '23', '34', '45', '56', '67', '78', '81']
        assert line_name in ['lhcb1', 'lhcb2']
        self.arc_name = arc_name
        self.line_name = line_name
        self.line = line

        beam_number = line_name[-1:]
        sector_start_number = arc_name[:1]
        sector_end_number = arc_name[1:]
        self.start_cell = f's.cell.{arc_name}.b{beam_number}'
        self.end_cell = f'e.cell.{arc_name}.b{beam_number}'
        self.start_arc = f'e.ds.r{sector_start_number}.b{beam_number}'
        self.end_arc = f's.ds.l{sector_end_number}.b{beam_number}'

    def run(self):

        twinit_cell = self.line.twiss(
                    start=self.start_cell, end=self.end_cell,
                    init='periodic', only_twiss_init=True)
        #  twinit_cell.element_name is start_cell for b1 and end_cell for b2

        tw_to_end_arc = self.line.twiss(init=twinit_cell,
            start=twinit_cell.element_name, end=self.end_arc)
        tw_to_start_arc = self.line.twiss(init=twinit_cell,
            start=self.start_arc,end=twinit_cell.element_name)

        mux_arc_from_cell = (tw_to_end_arc['mux', self.end_arc]
                             - tw_to_start_arc['mux', self.start_arc])
        muy_arc_from_cell = (tw_to_end_arc['muy', self.end_arc]
                             - tw_to_start_arc['muy', self.start_arc])

        return {
            'mux_arc_from_cell': mux_arc_from_cell,
            'muy_arc_from_cell': muy_arc_from_cell,
            'twinit_cell': twinit_cell,
            'tw_to_end_arc': tw_to_end_arc,
            'tw_to_start_arc': tw_to_start_arc}

class ActionMatchPhaseWithMQTs(xd.Action):

    def __init__(self, arc_name, line_name, line,
                 mux_arc_target, muy_arc_target, restore=False):

        self.action_arc_phase = ActionArcPhaseAdvanceFromCell(
            arc_name=arc_name, line_name=line_name, line=line)
        self.line = line
        self.mux_arc_target = mux_arc_target
        self.muy_arc_target = muy_arc_target
        self.restore = restore

        beam_number = line_name[-1:]
        self.mqt_knob_names = [
            f'kqtf.a{arc_name}b{beam_number}',
            f'kqtd.a{arc_name}b{beam_number}']

    def run(self):
        #store initial knob values
        mqt_knob_values = {
            kk: self.line.vars[kk]._value for kk in self.mqt_knob_names}

        opt = xd.Optimize(
            targets=[
                xd.Target(action=self.action_arc_phase, tar='mux_arc_from_cell',
                            value=self.mux_arc_target, tol=1e-8),
                xd.Target(action=self.action_arc_phase, tar='muy_arc_from_cell',
                            value=self.muy_arc_target, tol=1e-8),
            ],
            vary=[
                xd.VaryList(self.mqt_knob_names, self.line.vars, step=1e-5),
            ])
        opt.solve()

        res = {kk: np.abs(self.line.vars[kk]._value) for kk in self.mqt_knob_names}

        # restore initial knob values
        if self.restore:
            for kk in self.mqt_knob_names:
                self.line.vars[kk] = mqt_knob_values[kk]
        return res

@for_all_test_contexts
def test_match_nested(test_context):

    # Load the line
    collider = xtrack.Multiline.from_json(test_data_folder /
                'hllhc15_collider/collider_00_from_mad.json')
    collider.build_trackers(_context=test_context)

    collider.lhcb1.twiss_default['method'] = '4d'
    collider.lhcb2.twiss_default['method'] = '4d'
    collider.lhcb2.twiss_default['reverse'] = True

    action_arc_phase_s67_b1 = ActionArcPhaseAdvanceFromCell(
                        arc_name='67', line_name='lhcb1', line=collider.lhcb1)
    resb1 = action_arc_phase_s67_b1.run()

    action_arc_phase_s67_b2 = ActionArcPhaseAdvanceFromCell(
                        arc_name='67', line_name='lhcb2', line=collider.lhcb2)
    resb2 = action_arc_phase_s67_b2.run()

    # Check for b1
    twb1 = collider.lhcb1.twiss()
    mux_arc_target_b1 = twb1['mux', 's.ds.l7.b1'] - twb1['mux', 'e.ds.r6.b1']
    muy_arc_target_b1 = twb1['muy', 's.ds.l7.b1'] - twb1['muy', 'e.ds.r6.b1']
    xo.assert_allclose(resb1['mux_arc_from_cell'] , mux_arc_target_b1, rtol=1e-6)
    xo.assert_allclose(resb1['muy_arc_from_cell'] , muy_arc_target_b1, rtol=1e-6)

    # Check for b2
    twb2 = collider.lhcb2.twiss()
    mux_arc_target_b2 = twb2['mux', 's.ds.l7.b2'] - twb2['mux', 'e.ds.r6.b2']
    muy_arc_target_b2 = twb2['muy', 's.ds.l7.b2'] - twb2['muy', 'e.ds.r6.b2']
    xo.assert_allclose(resb2['mux_arc_from_cell'] , mux_arc_target_b2, rtol=1e-6)
    xo.assert_allclose(resb2['muy_arc_from_cell'] , muy_arc_target_b2, rtol=1e-6)

    starting_values = {
        'kqtf.a67b1': collider.vars['kqtf.a67b1']._value,
        'kqtf.a67b2': collider.vars['kqtf.a67b2']._value,
        'kqtd.a67b1': collider.vars['kqtd.a67b1']._value,
        'kqtd.a67b2': collider.vars['kqtd.a67b2']._value,
        'kqf.a67': collider.vars['kqf.a67']._value,
        'kqd.a67': collider.vars['kqd.a67']._value,
    }

    # # Perturb the quadrupoles
    collider.vars['kqtf.a67b1'] = starting_values['kqtf.a67b1'] * 1.1
    collider.vars['kqtf.a67b2'] = starting_values['kqtf.a67b2'] * 0.9
    collider.vars['kqtd.a67b1'] = starting_values['kqtd.a67b1'] * 0.15
    collider.vars['kqtd.a67b2'] = starting_values['kqtd.a67b2'] * 1.15

    collider.vars['kqd.a67'] = -0.00872
    collider.vars['kqf.a67'] = 0.00877

    action_match_mqt_s67_b1 = ActionMatchPhaseWithMQTs(
        arc_name='67', line_name='lhcb1', line=collider.lhcb1,
        mux_arc_target=mux_arc_target_b1, muy_arc_target=muy_arc_target_b1)
    action_match_mqt_s67_b2 = ActionMatchPhaseWithMQTs(
        arc_name='67', line_name='lhcb2', line=collider.lhcb2,
        mux_arc_target=mux_arc_target_b2, muy_arc_target=muy_arc_target_b2)


    optimize_phase_arc_67 = xd.Optimize(
        verbose=False,
        assert_within_tol=False,
        solver_options={'n_bisections': 3, 'min_step': 1e-5, 'n_steps_max': 5,},
        targets=[
            xd.Target(action=action_match_mqt_s67_b1, tar='kqtf.a67b1', value=0),
            xd.Target(action=action_match_mqt_s67_b1, tar='kqtd.a67b1', value=0),
            xd.Target(action=action_match_mqt_s67_b2, tar='kqtf.a67b2', value=0),
            xd.Target(action=action_match_mqt_s67_b2, tar='kqtd.a67b2', value=0),
        ],
        vary=[
            xd.Vary(name='kqf.a67', container=collider.vars, step=1e-5),
            xd.Vary(name='kqd.a67', container=collider.vars, step=1e-5),
        ])
    optimize_phase_arc_67.solve(n_steps=5)

    # Checks
    resb1_after = action_arc_phase_s67_b1.run()
    tw_init_arcb1 = resb1_after['tw_to_start_arc'].get_twiss_init('e.ds.r6.b1')
    twb1_after = collider.lhcb1.twiss(start='e.ds.r6.b1',
                                    end='s.ds.l7.b1',
                                    init=tw_init_arcb1)
    xo.assert_allclose(twb1_after['mux', 's.ds.l7.b1'] - twb1_after['mux', 'e.ds.r6.b1'],
                        mux_arc_target_b1, rtol=0, atol=1e-8)
    xo.assert_allclose(twb1_after['muy', 's.ds.l7.b1'] - twb1_after['muy', 'e.ds.r6.b1'],
                        muy_arc_target_b1, rtol=0, atol=1e-8)
    xo.assert_allclose(twb1_after['betx', 's.cell.67.b1'], twb1_after['betx', 'e.cell.67.b1'],
                        rtol=1e-8, atol=0)
    xo.assert_allclose(twb1_after['bety', 's.cell.67.b1'], twb1_after['bety', 'e.cell.67.b1'],
                        rtol=1e-8, atol=0)

    resb2_after = action_arc_phase_s67_b2.run()
    tw_init_arcb2 = resb2_after['tw_to_start_arc'].get_twiss_init('e.ds.r6.b2')
    twb2_after = collider.lhcb2.twiss(start='e.ds.r6.b2',
                                    end='s.ds.l7.b2',
                                    init=tw_init_arcb2)
    xo.assert_allclose(twb2_after['mux', 's.ds.l7.b2'] - twb2_after['mux', 'e.ds.r6.b2'],
                        mux_arc_target_b2, rtol=0, atol=1e-8)
    xo.assert_allclose(twb2_after['muy', 's.ds.l7.b2'] - twb2_after['muy', 'e.ds.r6.b2'],
                        muy_arc_target_b2, rtol=0, atol=1e-8)
    xo.assert_allclose(twb2_after['betx', 's.cell.67.b2'], twb2_after['betx', 'e.cell.67.b2'],
                        rtol=1e-8, atol=0)
    xo.assert_allclose(twb2_after['bety', 's.cell.67.b2'], twb2_after['bety', 'e.cell.67.b2'],
                        rtol=1e-8, atol=0)

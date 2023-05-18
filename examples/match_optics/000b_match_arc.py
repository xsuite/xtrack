
import numpy as np

import xtrack as xt

# xt._print.suppress = True

# Load the line
collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_collider/collider_00_from_mad.json')
collider.build_trackers()

collider.lhcb1.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['reverse'] = True

class ActionArcPhaseAdvanceFromCell(xt.Action):
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

    def compute(self):

        tw_cell_periodic = self.line.twiss(
                    ele_start=self.start_cell,
                    ele_stop=self.end_cell,
                    twiss_init='periodic')

        twinit_start_cell = tw_cell_periodic.get_twiss_init(self.start_cell)

        tw_to_end_arc = self.line.twiss(
            ele_start=self.start_cell,
            ele_stop=self.end_arc,
            twiss_init=twinit_start_cell)

        tw_to_start_arc = self.line.twiss(
            ele_start=self.start_arc,
            ele_stop=self.start_cell,
            twiss_init=twinit_start_cell)

        mux_arc_from_cell = (tw_to_end_arc['mux', self.end_arc]
                             - tw_to_start_arc['mux', self.start_arc])
        muy_arc_from_cell = (tw_to_end_arc['muy', self.end_arc]
                             - tw_to_start_arc['muy', self.start_arc])

        return {
            'mux_arc_from_cell': mux_arc_from_cell,
            'muy_arc_from_cell': muy_arc_from_cell,
            'tw_cell_periodic': tw_cell_periodic,
            'twinit_start_cell': twinit_start_cell,
            'tw_to_end_arc': tw_to_end_arc,
            'tw_to_start_arc': tw_to_start_arc}

action_arc_phase_s67_b1 = ActionArcPhaseAdvanceFromCell(
                    arc_name='67', line_name='lhcb1', line=collider.lhcb1)

res = action_arc_phase_s67_b1.compute()

tw = collider.lhcb1.twiss()
mux_arc_target = tw['mux', 's.ds.l7.b1'] - tw['mux', 'e.ds.r6.b1']
muy_arc_target = tw['muy', 's.ds.l7.b1'] - tw['muy', 'e.ds.r6.b1']

assert np.isclose(res['mux_arc_from_cell'] , mux_arc_target, rtol=1e-6)
assert np.isclose(res['muy_arc_from_cell'] , muy_arc_target, rtol=1e-6)
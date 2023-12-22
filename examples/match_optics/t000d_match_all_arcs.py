import time
import multiprocessing as mp
import pickle

import numpy as np

import xtrack as xt
import xdeps as xd

# xt._print.suppress = True



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

        if isinstance(self.line, xt.Multiline):
            line = self.line[self.line_name]

        twinit_cell = line.twiss(
                    start=self.start_cell, end=self.end_cell,
                    init='periodic', only_twiss_init=True)
        #  twinit_cell.element_name is start_cell for b1 and end_cell for b2

        tw_to_end_arc = line.twiss(init=twinit_cell,
            start=twinit_cell.element_name, end=self.end_arc)
        tw_to_start_arc = line.twiss(init=twinit_cell,
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

# For simplicity I use the same target for all arcs

def solve_optimizations(opts):
    for opt in opts:
        opt.solve()
        print(f'done optimization on {opt.vary[0].name}')

all_knobs = []
for arc in ['12', '23', '34', '45', '56', '67', '78', '81']:
    all_knobs += [f'kqf.a{arc}', f'kqd.a{arc}']
    all_knobs += [f'kqtf.a{arc}b1', f'kqtd.a{arc}b1']
    all_knobs += [f'kqtf.a{arc}b2', f'kqtd.a{arc}b2']

if __name__ == '__main__':

    # Load the line
    collider = xt.Multiline.from_json(
        '../../test_data/hllhc15_collider/collider_00_from_mad.json')
    collider.build_trackers()

    collider.lhcb1.twiss_default['method'] = '4d'
    collider.lhcb2.twiss_default['method'] = '4d'
    collider.lhcb2.twiss_default['reverse'] = True

    twb1 = collider.lhcb1.twiss()
    mux_arc_target_b1 = twb1['mux', 's.ds.l7.b1'] - twb1['mux', 'e.ds.r6.b1']
    muy_arc_target_b1 = twb1['muy', 's.ds.l7.b1'] - twb1['muy', 'e.ds.r6.b1']

    twb2 = collider.lhcb2.twiss()
    mux_arc_target_b2 = twb2['mux', 's.ds.l7.b2'] - twb2['mux', 'e.ds.r6.b2']
    muy_arc_target_b2 = twb2['muy', 's.ds.l7.b2'] - twb2['muy', 'e.ds.r6.b2']

    optimizations_to_do =[]

    for ss in ['12', '23', '34', '45', '56', '67', '78', '81']:

        action_match_mqt_s67_b1 = ActionMatchPhaseWithMQTs(
            arc_name=ss, line_name='lhcb1', line=collider,
            mux_arc_target=mux_arc_target_b1, muy_arc_target=muy_arc_target_b1)
        action_match_mqt_s67_b2 = ActionMatchPhaseWithMQTs(
            arc_name=ss, line_name='lhcb2', line=collider,
            mux_arc_target=mux_arc_target_b2, muy_arc_target=muy_arc_target_b2)

        optimize_phase_arc = xd.Optimize(
            verbose=False,
            assert_within_tol=False,
            solver_options={'n_bisections': 3, 'min_step': 1e-5, 'n_steps_max': 5,},
            targets=[
                xd.Target(action=action_match_mqt_s67_b1, tar=f'kqtf.a{ss}b1', value=0),
                xd.Target(action=action_match_mqt_s67_b1, tar=f'kqtd.a{ss}b1', value=0),
                xd.Target(action=action_match_mqt_s67_b2, tar=f'kqtf.a{ss}b2', value=0),
                xd.Target(action=action_match_mqt_s67_b2, tar=f'kqtd.a{ss}b2', value=0),
            ],
            vary=[
                xd.Vary(name=f'kqf.a{ss}', container=collider.vars, step=1e-5),
                xd.Vary(name=f'kqd.a{ss}', container=collider.vars, step=1e-5),
        ])

        optimizations_to_do.append(optimize_phase_arc)

    optimizations_to_do = 8 * [optimizations_to_do[0]]

    collider.vars.cache_active = True
    initial_values = {kk: collider.vars[kk]._value for kk in all_knobs} # create setters
    collider._var_sharing = None

    # t1 = time.time()
    # pool = mp.Pool(processes=4)
    # pool.map(solve_optimizations, [optimizations_to_do[:2], optimizations_to_do[2:4],
    #                                 optimizations_to_do[4:6], optimizations_to_do[6:]])
    # t2 = time.time()
    # print(f'Time spent(parallel): {t2-t1} s')

    t1 = time.time()
    for ii, opt in enumerate(optimizations_to_do):
        # restore initial values
        for kk in all_knobs:
            collider.vars[kk] = initial_values[kk]
        print('pickling...')
        tp1 = time.perf_counter()
        opt = pickle.loads(pickle.dumps(opt))
        tp2 = time.perf_counter()
        print(f'pickling time: {tp2-tp1} s')
        print('solving...')
        ts1 = time.perf_counter()
        opt.solve()
        ts2 = time.perf_counter()
        print(f'Optimization {ii} done in {ts2-ts1} s')
    t2 = time.time()
    print(f'Time spent (serial): {t2-t1} s')

    # xt.general._print.suppress = False
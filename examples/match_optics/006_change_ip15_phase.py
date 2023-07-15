import xtrack as xt
import lhc_match as lm

collider = xt.Multiline.from_json('hllhc.json')
collider.build_trackers()
collider.vars.load_madx_optics_file(
    "../../test_data/hllhc15_thick/opt_round_150_1500.madx")

d_mux_15_b1 = 0.1
d_muy_15_b1 = 0.12
d_mux_15_b2 = -0.09
d_muy_15_b2 = -0.15

# d_mux_15_b1 = 0
# d_muy_15_b1 = 0
# d_mux_15_b2 = 0
# d_muy_15_b2 = 0

arc_periodic_solution =lm.get_arc_periodic_solution(collider)

class ActionPhase_23_34(xt.Action):

    def __init__(self, collider):
        self.collider = collider

    def run(self):
        try:
            tw_arc = lm.get_arc_periodic_solution(self.collider, arc_name=['23', '34'])
        except ValueError:
            # Twiss failed
            return {
                'mux_23_34_b1': 1e100,
                'muy_23_34_b1': 1e100,
                'mux_23_34_b2': 1e100,
                'muy_23_34_b2': 1e100,
            }
        tw_23_b1 = tw_arc['lhcb1']['23']
        tw_23_b2 = tw_arc['lhcb2']['23']
        mux_23_b1 = tw_23_b1['mux', 's.ds.l3.b1'] - tw_23_b1['mux', 'e.ds.r2.b1']
        muy_23_b1 = tw_23_b1['muy', 's.ds.l3.b1'] - tw_23_b1['muy', 'e.ds.r2.b1']
        mux_23_b2 = tw_23_b2['mux', 's.ds.l3.b2'] - tw_23_b2['mux', 'e.ds.r2.b2']
        muy_23_b2 = tw_23_b2['muy', 's.ds.l3.b2'] - tw_23_b2['muy', 'e.ds.r2.b2']

        tw34_b1 = tw_arc['lhcb1']['34']
        tw34_b2 = tw_arc['lhcb2']['34']
        mux_34_b1 = tw34_b1['mux', 's.ds.l4.b1'] - tw34_b1['mux', 'e.ds.r3.b1']
        muy_34_b1 = tw34_b1['muy', 's.ds.l4.b1'] - tw34_b1['muy', 'e.ds.r3.b1']
        mux_34_b2 = tw34_b2['mux', 's.ds.l4.b2'] - tw34_b2['mux', 'e.ds.r3.b2']
        muy_34_b2 = tw34_b2['muy', 's.ds.l4.b2'] - tw34_b2['muy', 'e.ds.r3.b2']

        return {
            'mux_23_34_b1': mux_23_b1 + mux_34_b1,
            'muy_23_34_b1': muy_23_b1 + muy_34_b1,
            'mux_23_34_b2': mux_23_b2 + mux_34_b2,
            'muy_23_34_b2': muy_23_b2 + muy_34_b2
        }

action_phase_23_34 = ActionPhase_23_34(collider)
phase_23_34_0 = action_phase_23_34.run()

d_mux_15_b1_target = phase_23_34_0['mux_23_34_b1'] + d_mux_15_b1
d_muy_15_b1_target = phase_23_34_0['muy_23_34_b1'] + d_muy_15_b1
d_mux_15_b2_target = phase_23_34_0['mux_23_34_b2'] + d_mux_15_b2
d_muy_15_b2_target = phase_23_34_0['muy_23_34_b2'] + d_muy_15_b2

opt = collider.match(
    solve=False,
    targets=[
        action_phase_23_34.target('mux_23_34_b1', d_mux_15_b1_target),
        action_phase_23_34.target('muy_23_34_b1', d_muy_15_b1_target),
        action_phase_23_34.target('mux_23_34_b2', d_mux_15_b2_target),
        action_phase_23_34.target('muy_23_34_b2', d_muy_15_b2_target),
    ],
    vary=xt.VaryList(['kqf.a23', 'kqd.a23', 'kqf.a34', 'kqd.a34'])
)

opt.solve()
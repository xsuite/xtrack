import xtrack as xt
import lhc_match as lm

collider = xt.Multiline.from_json('hllhc.json')
collider.build_trackers()
collider.vars.load_madx_optics_file(
    "../../test_data/hllhc15_thick/opt_round_150_1500.madx")


arc_name = '67'
target_mux_b1 = collider.varval['mux67b1']
target_muy_b1 = collider.varval['muy67b1']
target_mux_b2 = collider.varval['mux67b2']
target_muy_b2 = collider.varval['muy67b2']

action_phase_b1 = lm.ActionArcPhaseAdvanceFromCell(
                    collider=collider, line_name='lhcb1', arc_name=arc_name)
action_phase_b2 = lm.ActionArcPhaseAdvanceFromCell(
                    collider=collider, line_name='lhcb2', arc_name=arc_name)

opt=collider.match(
    solve=False,
    targets=[
        action_phase_b1.target('mux', target_mux_b1),
        action_phase_b1.target('muy', target_muy_b1),
        action_phase_b2.target('mux', target_mux_b2),
        action_phase_b2.target('muy', target_muy_b2),
    ],
    vary=[
        xt.VaryList([f'kqtf.a{arc_name}b1', f'kqtd.a{arc_name}b1',
                     f'kqtf.a{arc_name}b2', f'kqtd.a{arc_name}b2',
                     f'kqf.a{arc_name}', f'kqd.a{arc_name}'
                     ]),
    ])
opt.solve()
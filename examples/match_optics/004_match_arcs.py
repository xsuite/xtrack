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

starting_values = {
    'kqtf.a67b1': collider.vars['kqtf.a67b1']._value,
    'kqtf.a67b2': collider.vars['kqtf.a67b2']._value,
    'kqtd.a67b1': collider.vars['kqtd.a67b1']._value,
    'kqtd.a67b2': collider.vars['kqtd.a67b2']._value,
    'kqf.a67': collider.vars['kqf.a67']._value,
    'kqd.a67': collider.vars['kqd.a67']._value,
}

# Perturb the quadrupoles
collider.vars['kqtf.a67b1'] = starting_values['kqtf.a67b1'] * 1.1
collider.vars['kqtf.a67b2'] = starting_values['kqtf.a67b2'] * 0.9
collider.vars['kqtd.a67b1'] = starting_values['kqtd.a67b1'] * 0.15
collider.vars['kqtd.a67b2'] = starting_values['kqtd.a67b2'] * 1.15
collider.vars['kqd.a67'] = -0.00872
collider.vars['kqf.a67'] = 0.00877

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
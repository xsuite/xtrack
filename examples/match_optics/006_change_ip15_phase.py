import xtrack as xt
import lhc_match as lm

collider = xt.Multiline.from_json('hllhc.json')
collider.build_trackers()
collider.vars.load_madx_optics_file(
    "../../test_data/hllhc15_thick/opt_round_150_1500.madx")

d_mux_15_b1 = None
d_muy_15_b1 = None
d_mux_15_b2 = None
d_muy_15_b2 = None

d_mux_15_b1 = 0.1
d_muy_15_b1 = 0.12
# d_mux_15_b2 = -0.09
# d_muy_15_b2 = -0.15

action_phase_23_34 = lm.ActionPhase_23_34(collider)
action_phase_67_78 = lm.ActionPhase_67_78(collider)

phase_23_34_0 = action_phase_23_34.run()
phase_67_78_0 = action_phase_67_78.run()

mux_23_34_b1_target = phase_23_34_0['mux_23_34_b1']
muy_23_34_b1_target = phase_23_34_0['muy_23_34_b1']
mux_23_34_b2_target = phase_23_34_0['mux_23_34_b2']
muy_23_34_b2_target = phase_23_34_0['muy_23_34_b2']
mux_67_78_b1_target = phase_67_78_0['mux_67_78_b1']
muy_67_78_b1_target = phase_67_78_0['muy_67_78_b1']
mux_67_78_b2_target = phase_67_78_0['mux_67_78_b2']
muy_67_78_b2_target = phase_67_78_0['muy_67_78_b2']

n_contraints = 0
targets = []
if d_mux_15_b1 is not None:
    mux_23_34_b1_target += d_mux_15_b1
    mux_67_78_b1_target -= d_mux_15_b1
    targets.append(action_phase_23_34.target('mux_23_34_b1', mux_23_34_b1_target))
    targets.append(action_phase_67_78.target('mux_67_78_b1', mux_67_78_b1_target))
    n_contraints += 1

if d_muy_15_b1 is not None:
    muy_23_34_b1_target += d_muy_15_b1
    muy_67_78_b1_target -= d_muy_15_b1
    targets.append(action_phase_23_34.target('muy_23_34_b1', muy_23_34_b1_target))
    targets.append(action_phase_67_78.target('muy_67_78_b1', muy_67_78_b1_target))
    n_contraints += 1

if d_mux_15_b2 is not None:
    mux_23_34_b2_target += d_mux_15_b2
    mux_67_78_b2_target -= d_mux_15_b2
    targets.append(action_phase_23_34.target('mux_23_34_b2', mux_23_34_b2_target))
    targets.append(action_phase_67_78.target('mux_67_78_b2', mux_67_78_b2_target))
    n_contraints += 1

if d_muy_15_b2 is not None:
    muy_23_34_b2_target += d_muy_15_b2
    muy_67_78_b2_target -= d_muy_15_b2
    targets.append(action_phase_23_34.target('muy_23_34_b2', muy_23_34_b2_target))
    targets.append(action_phase_67_78.target('muy_67_78_b2', muy_67_78_b2_target))
    n_contraints += 1

vary = [
    xt.VaryList(['kqf.a23', 'kqd.a23', 'kqf.a34', 'kqd.a34'], weight=5),
    xt.VaryList(['kqf.a67', 'kqd.a67', 'kqf.a78', 'kqd.a78'], weight=5),
]
if n_contraints > 2:
    vary += [
        xt.VaryList(['kqtf.a23b1', 'kqtd.a23b1', 'kqtf.a34b1', 'kqtd.a34b1',
                    'kqtf.a23b2', 'kqtd.a23b2', 'kqtf.a34b2', 'kqtd.a34b2']),
        xt.VaryList(['kqtf.a67b1', 'kqtd.a67b1', 'kqtf.a78b1', 'kqtd.a78b1',
                    'kqtf.a67b2', 'kqtd.a67b2', 'kqtf.a78b2', 'kqtd.a78b2']),
    ]

opt = collider.match(
    solve=False,
    solver_options={'n_bisections': 5},
    vary=vary,
    targets=targets,
)

opt.solve()
import xtrack as xt

import xtrack._temp.lhc_match as lm

collider = xt.Multiline.from_json('hllhc.json')
collider.build_trackers()
collider.vars.load_madx_optics_file(
    "../../test_data/hllhc15_thick/opt_round_150_1500.madx")

for aa in lm.ARC_NAMES:
    print(f"Matching arc {aa}")
    target_mux_b1 = collider.varval[f'mux{aa}b1']
    target_muy_b1 = collider.varval[f'muy{aa}b1']
    target_mux_b2 = collider.varval[f'mux{aa}b2']
    target_muy_b2 = collider.varval[f'muy{aa}b2']

    opt = lm.match_arc_phase_advance(collider=collider, arc_name=aa,
                    target_mux_b1=target_mux_b1, target_muy_b1=target_muy_b1,
                    target_mux_b2=target_mux_b2, target_muy_b2=target_muy_b2)

# Check
for aa in lm.ARC_NAMES:
    collider.varval[f'kqtf.a{aa}b1'] *= 1.1
    collider.varval[f'kqtf.a{aa}b2'] *=  0.9
    collider.varval[f'kqtd.a{aa}b1'] *= 0.15
    collider.varval[f'kqtd.a{aa}b2'] *= 1.15
    collider.varval[f'kqd.a{aa}'] *= 1.03
    collider.varval[f'kqf.a{aa}'] *= 1.02

for aa in lm.ARC_NAMES:
    print(f"Matching arc {aa}")
    target_mux_b1 = collider.varval[f'mux{aa}b1']
    target_muy_b1 = collider.varval[f'muy{aa}b1']
    target_mux_b2 = collider.varval[f'mux{aa}b2']
    target_muy_b2 = collider.varval[f'muy{aa}b2']

    opt = lm.match_arc_phase_advance(collider=collider, arc_name=aa,
                    target_mux_b1=target_mux_b1, target_muy_b1=target_muy_b1,
                    target_mux_b2=target_mux_b2, target_muy_b2=target_muy_b2)
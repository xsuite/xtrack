import xtrack as xt
import lhc_match as lm

collider = xt.Multiline.from_json('hllhc.json')
collider.build_trackers()
collider.vars.load_madx_optics_file(
    "../../test_data/hllhc15_thick/opt_round_150_1500.madx")

for arc_name in lm.ARC_NAMES:
    print(f"Matching arc {arc_name}")
    target_mux_b1 = collider.varval[f'mux{arc_name}b1']
    target_muy_b1 = collider.varval[f'muy{arc_name}b1']
    target_mux_b2 = collider.varval[f'mux{arc_name}b2']
    target_muy_b2 = collider.varval[f'muy{arc_name}b2']

    opt = lm.match_arc_phase_advance(collider=collider, arc_name=arc_name,
                        target_mux_b1=target_mux_b1, target_muy_b1=target_muy_b1,
                        target_mux_b2=target_mux_b2, target_muy_b2=target_muy_b2)
# Need to:
# - have indepedent coupling knobs for the two beams
# - have independent octupole knobs for the two beams
# - add knobs for optics correction


import json

import numpy as np
import xtrack as xt
import xobjects as xo

with open('../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json',
            'r') as fid:
    dct_b1 = json.load(fid)
line = xt.Line.from_dict(dct_b1)

beamn = 1
line.vars[f'c_minus_re_b{beamn}'] = 0
line.vars[f'c_minus_im_b{beamn}'] = 0
for ii in [1, 2, 3, 4, 5, 6, 7, 8]:
    for jj, nn in zip([1, 2], ['re', 'im']):
        old_name = f'b{ii}{jj}'
        new_name = f'coeff_skew_{ii}{jj}_b{beamn}'
        line.vars[new_name] = line.vars[old_name]._value

        # Identify controlled circuit
        targets = line.vars[old_name]._find_dependant_targets()
        if len(targets) > 1: # Controls something
            ttt = [t for t in targets if repr(t).startswith('vars[') and
                   repr(t) != f"vars['{old_name}']"]
            assert len(ttt) > 0
            assert len(ttt) < 3

            for kqs_knob in ttt:
                kqs_knob_str = repr(kqs_knob)
                assert "'" in kqs_knob_str
                assert '"' not in kqs_knob_str
                var_name = kqs_knob_str.split("'")[1]
                assert var_name.startswith('kqs')
                line.vars[var_name] += (line.vars[new_name]
                            * line.vars[f'c_minus_{nn}_b{beamn}'])


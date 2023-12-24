import numpy as np
import xtrack as xt

# Load a line and build a tracker
line = xt.Line.from_json('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.build_tracker()

class ActionMeasAmplDet(xt.Action):

    def __init__(self, line, num_turns, nemitt_x, nemitt_y):

        self.line = line
        self.num_turns = num_turns
        self.nemitt_x = nemitt_x
        self.nemitt_y = nemitt_y

    def run(self):

        det_coefficients = self.line.get_amplitude_detuning_coefficients(
                                                    num_turns=self.num_turns)
        return det_coefficients

action = ActionMeasAmplDet(line=line, nemitt_x=2.5e-6, nemitt_y=2.5e-6,
                           num_turns=128)

opt = line.match(vary=xt.VaryList(['kof.a23b1', 'kod.a23b1'], step=1.),
                 targets=[action.target('det_xx', 1000., tol=0.1),
                          action.target('det_yy', 2000., tol=0.1)])

opt.target_status()
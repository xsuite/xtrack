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
                                nemitt_x=self.nemitt_x, nemitt_y=self.nemitt_y,
                                num_turns=self.num_turns)

        out = {'d_xx': det_coefficients['det_xx'],
               'd_yy': det_coefficients['det_yy']}

        return out

action = ActionMeasAmplDet(line=line, nemitt_x=2.5e-6, nemitt_y=2.5e-6,
                           num_turns=128)

opt = line.match(vary=xt.VaryList(['kof.a23b1', 'kod.a23b1'], step=1.),
                 targets=[action.target('d_xx', 1000., tol=0.1),
                          action.target('d_yy', 2000., tol=0.1)])

opt.target_status()
# prints:
#
# Target status:
# id state tag tol_met     residue current_val target_val description
#  0 ON           True   0.0844456     1000.08       1000 'd_xx', val=1000, ...
#  1 ON           True -0.00209987        2000       2000 'd_yy', val=2000, ...
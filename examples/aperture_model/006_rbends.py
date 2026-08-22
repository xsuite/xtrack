import xtrack as xt
from xtrack.aperture.structures import SurveyData
import numpy as np
import matplotlib.pyplot as plt

angle = np.pi / 6
angle_diff = -angle
shift_in = 0
model = 'straight-body'

env = xt.Environment()
dr_ = env.new('dr', 'Drift', length=10)
rb_ = env.new('rb', 'RBend', length_straight=10, angle=angle, rbend_angle_diff=angle_diff, rbend_model=model)

rb = env['rb']

rb.rbend_compensate_sagitta = False
rbend_shift = shift_in + (np.cos(rb._angle_in) - np.sqrt(1 - (np.sin(rb._angle_in) - 0.5 * rb.h * rb.length_straight) ** 2)) / rb.h
rb.rbend_shift = rbend_shift

ll = env.new_line(components=[dr_, rb_, dr_])
sv_thick = ll.survey()

cuts = np.linspace(0, 30, 37)

ls = ll.copy()
ls.cut_at_s(cuts)
sv_sliced = ls.survey()

sv_xo = SurveyData.from_survey_table(sv_thick, ll)
sv_resampled = sv_xo.resample(cuts)
X_resampled, Z_resampled = sv_resampled.pose.to_nparray()[:, (0, 2), 3].T

# Plot it all
fig = plt.figure()
ax = fig.add_subplot()

ax.plot(sv_thick.Z, sv_thick.X, label='thick', marker='o')
ax.plot(sv_sliced.Z, sv_sliced.X, label='sliced', marker='+')
ax.plot(Z_resampled, X_resampled, label='resampled', marker='x')

ax.legend()
plt.show()

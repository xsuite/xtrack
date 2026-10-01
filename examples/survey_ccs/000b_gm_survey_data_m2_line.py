import xtrack as xt
import numpy as np

from xtrack._temp import survey_utils as su

compensate_psi_vbend = True
psi_tol_deg = 20

env = xt.load('M2_MTN3p8_v6_notilt.seq')
line = env['m2']

# env = xt.load(['ldb/LS3.seq'])
# line = env['m2']

# Define T09 survey parameters.
X0, Y0, Z0 = -677.24488, 2441.56985, 4605.8619
theta0, phi0, psi0 = 1.406046 - ((np.pi) / 2), -0.000358, 0

sv = line.survey(include_element_frames=True,
                 X0=X0, Y0=Y0, Z0=Z0,
                 theta0=theta0, phi0=phi0, psi0=psi0)

# List of elements requiring alignment data
names_align = sv.rows.match_not(name='.*drift.*|.*_aper|_end_point')\
            .rows.match_not(element_type='Limit.*|Translation|Rotation').name

su.write_legacy_survey_tfs(
    'survey_output.tfs',
    survey=sv,
    element_names=names_align,
    element_container=env,
    compensate_psi_vbend=compensate_psi_vbend,
    psi_tol_deg=psi_tol_deg
)


# Same format, but with the points on the reference trajectory and the
# angle and tilt of the elements (compatible with the MAD-X survey output)
su.write_legacy_survey_tfs(
    'survey_output_ref_trajectory.tfs',
    survey=sv,
    element_names=names_align,
    element_container=env,
    reference_trajectory=True
)

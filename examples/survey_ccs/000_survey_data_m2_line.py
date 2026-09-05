import xtrack as xt
import numpy as np

from xtrack._temp import survey_utils as su

env = xt.load(['M2_with_jump.seq', 'M2_MTN3p8_v5_notilt.str'])
line = env['m2']

# Define T09 survey parameters.
X0, Y0, Z0 = -677.24488, 2441.56985, 4605.8619
theta0, phi0, psi0 = 1.406046 - ((np.pi) / 2), -0.000358, 0

sv = line.survey(include_element_frames=True,
                 X0=X0, Y0=Y0, Z0=Z0,
                 theta0=theta0, phi0=phi0, psi0=psi0)

# List of elements requiring alignment data
names_align = sv.rows.match_not(name='.*drift.*|.*_aper|_end_point')\
            .rows.match_not(element_type='Limit.*|Translation|Rotation').name

# Extract absolute coordinates of start/end of all elements
frames = {}
for nn in names_align:
    frames[nn] = sv.get_all_frames(nn)

# Prepare output table with CCS start/end
name = []
x_ccs = []
y_ccs = []
z_ccs = []
theta_gon_ccs = []
phi_rad_ccs = []
psi_rad_ccs = []

for nn in names_align:
    p_start_ccs = frames[nn]['elem_start'].to_ccs()
    p_end_ccs = frames[nn]['elem_end'].to_ccs()

    name.append(nn+'.u')
    x_ccs.append(p_start_ccs.x)
    y_ccs.append(p_start_ccs.y)
    z_ccs.append(p_start_ccs.z)
    theta_gon_ccs.append(p_start_ccs.theta_gon)
    phi_rad_ccs.append(p_start_ccs.phi)
    psi_rad_ccs.append(p_start_ccs.psi)

    name.append(nn+'.d')
    x_ccs.append(p_end_ccs.x)
    y_ccs.append(p_end_ccs.y)
    z_ccs.append(p_end_ccs.z)
    theta_gon_ccs.append(p_end_ccs.theta_gon)
    phi_rad_ccs.append(p_end_ccs.phi)
    psi_rad_ccs.append(p_end_ccs.psi)

dct_out = {
    'name': name,
    'x_ccs': x_ccs,
    'y_ccs': y_ccs,
    'z_ccs': z_ccs,
    'theta_gon_ccs': theta_gon_ccs,
    'phi_rad_ccs': phi_rad_ccs,
    'psi_rad_ccs': psi_rad_ccs
}

for kk in dct_out:
    dct_out[kk] = np.array(dct_out[kk])

dct_out['theta_rad'] = dct_out['theta_gon_ccs'] * (np.pi / 200)
tt_out = xt.Table(dct_out)

# Save the table in csv and tfs formats
tt_out.to_csv('m2_ccs_align.csv')
tt_out.to_tfs('m2_ccs_align.tfs', float_precision=15)

# Prepare a tfs file compatible with import in GEODE
# - Only selected elements are included in the GEODE-compatible survey file.
# - Point associated with each element name is the actual position of the
#  element's end point.
# - Position of the element's start point is given in the dummy drift before the
#  element (such drifts have zero length)
# - All elements are defined as straight (theta, phi, psi, represent the orientation
#  of the physical axis of the element.
su.write_legacy_survey_tfs(
    'survey_output.tfs',
    survey=sv,
    element_names=names_align,
    element_container=env,
)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(figsize=(10, 6))
ax1 = plt.subplot(2, 1, 1)
plt.plot(sv.Z, sv.X)
plt.xlabel('Z [m]')
plt.ylabel('X [m]')

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(sv.Z, sv.Y)
plt.xlabel('Z [m]')
plt.ylabel('Y [m]')

for nn in names_align:
    p_start = frames[nn]['elem_start']
    p_end = frames[nn]['elem_end']
    ax1.plot([p_start.Z, p_end.Z], [p_start.X, p_end.X], '.-r')
    ax2.plot([p_start.Z, p_end.Z], [p_start.Y, p_end.Y], '.-r')
plt.suptitle('Survey of M2 line')
plt.show()

import xtrack as xt
import numpy as np

# env = xt.load(['M2_with_jump.seq', 'M2_MTN3p8_v5_notilt.str'])
# line = env['m2']

env = xt.load(['ldb/LS3.seq'])
line = env['m2']

# Define T09 survey parameters.
X0, Y0, Z0 = -677.24488, 2441.56985, 4605.8619
theta0, phi0, psi0 = 1.406046 - ((np.pi) / 2), -0.000358, 0

sv = line.survey(include_element_frames=True,
                 X0=X0, Y0=Y0, Z0=Z0,
                 theta0=theta0, phi0=phi0, psi0=psi0)

# List of elements requiring alignment data
names_align = sv.rows.match_not(name='.*drift.*|.*_aper|_end_point')\
            .rows.match_not(element_type='Limit*|Translation|Rotation').name

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

tt_out.to_csv('m2_ccs_align.csv')
tt_out.to_tfs('m2_ccs_align.tfs', float_precision=15)

header = """
@ NAME             %06s "SURVEY"
@ TYPE             %06s "SURVEY"
@ TITLE            %08s "no-title"
@ ORIGIN           %17s "5.09.03 Darwin 64"
@ DATE             %08s "02/09/26"
@ TIME             %08s "12.10.29"
* NAME               KEYWORD                                    S                         L                     ANGLE                         X                         Y                         Z                     THETA                       PHI                       PSI                GLOBALTILT                      TILT    SLOT_ID ASSEMBLY_ID                  MECH_SEP                     V_POS
$ %s                 %s                                       %le                       %le                       %le                       %le                       %le                       %le                       %le                       %le                       %le                       %le                       %le         %d         %d                       %le                       %le
"""
header = header.lstrip()

lines = []
for ii, nn in enumerate(names_align):
    ff = sv.get_all_frames(nn)
    fthis = ff['elem_end']
    for place in ['start', 'end']:

        if place == 'start':
            fthis = ff['elem_start']
            this_name = f'drift_{ii}'
            this_s = sv['s', nn]
            this_length = np.linalg.norm(ff['elem_end'].XYZ - ff['elem_start'].XYZ, 2)
            this_slot_id = 0
        else:
            fthis = ff['elem_end']
            this_name = nn
            this_s = sv['s', nn + '>>1']
            this_length = 0
            this_slot_id = env[nn].extra['slot_id']

        this_line = ' '

        # NAME
        name_str = '"' + this_name.upper() + '"'
        name_str = name_str.ljust(20)
        this_line += name_str

        # KEYWORD
        keyword_str = '"' + 'ELEMENT' + '"'
        keyword_str = keyword_str.ljust(20)
        this_line += keyword_str

        # S (need to handle if entry or exit)
        val = this_s
        val_str = f"{val:26.9f}"
        this_line += val_str

        # L
        val = this_length
        val_str = f"{val:26.9f}"
        this_line += val_str

        # ANGLE
        val = 0
        val_str = f"{val:26.9f}"
        this_line += val_str

        # X
        val = fthis.X
        val_str = f"{val:26.9f}"
        this_line += val_str

        # Y
        val = fthis.Y
        val_str = f"{val:26.9f}"
        this_line += val_str

        # Z
        val = fthis.Z
        val_str = f"{val:26.9f}"
        this_line += val_str

        # THETA
        val = fthis.theta
        val_str = f"{val:26.9f}"
        this_line += val_str

        # PHI
        val = fthis.phi
        val_str = f"{val:26.9f}"
        this_line += val_str

        # PSI
        val = fthis.psi
        val_str = f"{val:26.9f}"
        this_line += val_str

        # GLOBALTILT
        val = fthis.psi
        val_str = f"{val:26.9f}"
        this_line += val_str

        # TILT
        val = 0
        val_str = f"{val:26.9f}"
        this_line += val_str

        # SLOT_ID
        val = this_slot_id
        val_str = f"{int(val):11d}"
        this_line += val_str

        # ASSEMBLY_ID
        val = 0
        val_str = f"{val:11d}"
        this_line += val_str

        # MECH_SEP
        val = 0
        val_str = f"{val:26.9f}"
        this_line += val_str

        # V_POS
        val = 0
        val_str = f"{val:26.9f}"
        this_line += val_str

        # # COMMENT
        # val = ' ""'
        # val_str = val.ljust(20)
        # this_line += val_str

        lines.append(this_line)

out = header
out += '\n'.join(lines)
with open('survey_output.tfs', 'w') as f:
    f.write(out)















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
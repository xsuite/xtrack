from cpymad.madx import Madx
import xtrack as xt

env = xt.load(['M2_with_jump.seq', 'M2_MTN3p8_v5_notilt.str'])
line = env['m2']

sv = line.survey(include_element_frames=True)

# List of elements requiring alignment data
names_align = sv.rows.match_not(name='.*drift.*|.*_aper|_end_point')\
            .rows.match_not(element_type='Marker|Limit*|Translation|Rotation').name

# Extract absolute coordinates of start/end of all elements
frames = {}
for nn in names_align:
    frames[nn] = sv.get_all_frames(nn)

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
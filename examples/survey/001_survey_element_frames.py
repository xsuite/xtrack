# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np
import xtrack as xt

# Build a line containing a misaligned quadrupole.
env = xt.Environment()

line = env.new_line(length=10, components=[
    env.new('q', xt.Quadrupole, length=4, at=5),
])

env['q'].rot_shift_anchor = 1.0  # Misalignment pivot, from the element start
env['q'].rot_y_rad = np.deg2rad(30)
env['q'].shift_x = 0.1

# Include the reference and actual element frames in the survey table.
survey = line.survey(include_element_frames=True)

# Inspect the element entrance and exit coordinates.
survey.cols[
    'name s '
    'X_elem_start Y_elem_start Z_elem_start '
    'X_elem_end Y_elem_end Z_elem_end'
]

# Extract all four frames associated with the quadrupole.
frames_at_q = survey.get_all_frames('q')
q_start = frames_at_q['elem_start']
q_end = frames_at_q['elem_end']
q_ref_start = frames_at_q['ref_start']
q_ref_end = frames_at_q['ref_end']

# Frames expose their position, local unit vectors, and survey angles.
q_start.XYZ
q_start.es
q_start.theta

# Frames can also be expressed in the CERN Coordinate System (CCS).
q_start_ccs = q_start.to_ccs()

# Plot the reference trajectory and the actual quadrupole position.
import matplotlib.pyplot as plt
plt.close('all')

fig1, ax = plt.subplots(figsize=(6.4, 4.8))
ax.plot(survey.Z, survey.X, '.-', label='Reference trajectory')
ax.plot(
    [q_ref_start.Z, q_ref_end.Z],
    [q_ref_start.X, q_ref_end.X],
    linewidth=5,
    alpha=0.35,
    label='Quadrupole reference placement',
)
ax.plot(
    [q_start.Z, q_end.Z],
    [q_start.X, q_end.X],
    '.-',
    linewidth=2,
    label='Actual quadrupole position',
)
ax.set_xlabel('Z [m]')
ax.set_ylabel('X [m]')
ax.set_title('Reference and element frames')
ax.axis('equal')
ax.grid(True, alpha=0.3)
ax.legend()
fig1.subplots_adjust(left=0.13, right=0.97, bottom=0.13, top=0.90)
plt.show()

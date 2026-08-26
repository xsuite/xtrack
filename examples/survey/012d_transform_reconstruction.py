import xtrack as xt
import numpy as np

env = xt.Environment()
env.set_particle_ref('proton', p0c=400e9)

line1 = env.new_line(length=15, components=[
    env.new('q1', 'Bend', length=2, angle=0.2, k1=0.1, at=5,
            rot_s_rad=np.deg2rad(30),
            rot_s_rad_no_frame=np.deg2rad(10),
            shift_x=0.3, rot_y_rad=np.deg2rad(10)),
])

sv1 = line1.survey(include_element_frames=True)

line2 = env.new_line(length=15, components=[
    env.new('translation', 'Translation', at=2, shift_x=0.4),
    env.new('rotation', 'Rotation', at=2,
            rot_y_rad=np.deg2rad(15), rot_x_rad=np.deg2rad(20), rot_s_rad=np.deg2rad(30)),
    env.new('q2', 'Bend', length=2, angle=0.2, k1=0.1, at=5)
])

sv2 = line2.survey(include_element_frames=True)

p1_mat_elem_start = np.eye(4)
p1_mat_elem_start[:3, :3] = sv1['E_elem_start', 'q1']
p1_mat_elem_start[:3, 3] = sv1['XYZ_elem_start', 'q1']

p2_mat_ref_start = np.eye(4)
p2_mat_ref_start[:3, :3] = sv2['E_ref_start', 'q2']
p2_mat_ref_start[:3, 3] = sv2['XYZ_ref_start', 'q2']

A = np.linalg.inv(p2_mat_ref_start) @ p1_mat_elem_start

theta = np.arctan2(A[0, 2], A[2, 2])
phi = np.arctan2(A[1, 2], np.sqrt(A[1, 0]**2 + A[1, 1]**2))
psi = np.arctan2(A[1, 0], A[1, 1])
dx = A[0, 3]
dy = A[1, 3]
ds = A[2, 3]

env['q2'].rot_y_rad = theta
env['q2'].rot_x_rad = phi
env['q2'].rot_s_rad = env['q1'].rot_s_rad
env['q2'].rot_s_rad_no_frame = psi - env['q1'].rot_s_rad
env['q2'].shift_x = dx
env['q2'].shift_y = dy
env['q2'].shift_s = ds

sv2_reconstructed = line2.survey(include_element_frames=True)

print('---- Starting point ----')
print('Original q1 frame:')
print(sv1['E_elem_start', 'q1'])
print(sv1['XYZ_elem_start', 'q1'])
print('Reconstructed q2 frame:')
print(sv2_reconstructed['E_elem_start', 'q2'])
print(sv2_reconstructed['XYZ_elem_start', 'q2'])

print('---- Ending point ----')
print('Original q1 frame:')
print(sv1['E_elem_end', 'q1'])
print(sv1['XYZ_elem_end', 'q1'])
print('Reconstructed q2 frame:')
print(sv2_reconstructed['E_elem_end', 'q2'])
print(sv2_reconstructed['XYZ_elem_end', 'q2'])

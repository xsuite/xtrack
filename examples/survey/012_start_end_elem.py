import xtrack as xt
import misalignment_entry_transforms as met
import numpy as np
import matplotlib.pyplot as plt

env = xt.Environment()

env.new('q', 'Quadrupole', length=2, k1=0.1,
        rot_shift_anchor=1, rot_y_rad=np.deg2rad(30))
env.new('b', 'Bend', length=2, angle=0, #np.deg2rad(0.1),
        rot_shift_anchor=1, rot_y_rad=np.deg2rad(30))
line_thick = env.new_line(length=8, components=[
    env.place('q', at=2),
    env.place('b', at=6),
])

line_sliced = line_thick.copy(shallow=True)
line_sliced.cut_at_s(np.linspace(0, 8, 33))

# line = line_sliced
# name = 'q..3'

# line = line_thick
# name = 'q'

line = line_thick
name = 'b'

sv = line.survey()

elem = line[name]

# Slices inherit their transformations and geometry from the parent element.
# This mirrors GET_PARAM, GET_WEIGHT, and the slice-anchor correction in
# track_local_particle_with_transformations.h.
parent = getattr(elem, '_parent', None)
if parent is None:
    element_with_transformations = elem
    weight = 1.0
    slice_offset = 0.0
else:
    element_with_transformations = parent
    weight = elem.weight
    slice_offset = elem.slice_offset

transform_kwargs = dict(
    shift_x=element_with_transformations.shift_x,
    shift_y=element_with_transformations.shift_y,
    shift_s=element_with_transformations.shift_s,
    rot_y_rad=element_with_transformations.rot_y_rad,
    rot_x_rad=element_with_transformations.rot_x_rad,
    rot_s_rad_no_frame=(
        element_with_transformations.rot_s_rad_no_frame),
    rot_shift_anchor=(
        element_with_transformations.rot_shift_anchor - slice_offset),
    length=getattr(element_with_transformations, 'length', 0.0) * weight,
    angle=getattr(element_with_transformations, 'angle', 0.0) * weight,
    h=getattr(element_with_transformations, 'h', 0.0),
    rot_s_rad=element_with_transformations.rot_s_rad,
)

transf_params_start = met.get_entry_transform(
    **transform_kwargs)

transf_params_end = met.get_exit_transform(
    **transform_kwargs)


XYZ_ref_start = sv['XYZ', name]
E_ref_start = sv['E_matrix', name]

XYZ_ref_end = sv['XYZ', name+'>>1']
E_ref_end = sv['E_matrix', name+'>>1']

##### ref_start -> elem_start

trans_start = xt.Translation(
    shift_x=transf_params_start.shift_x,
    shift_y=transf_params_start.shift_y,
)
rot_start = xt.Rotation(
    rot_x_rad=-(transf_params_start.rot_x_rad),
    rot_y_rad=transf_params_start.rot_y_rad,
    rot_s_rad=transf_params_start.rot_s_rad_no_frame,
    seq='yxs'
)

# shift by trans_start
XYZ_temp, E_elem_temp = trans_start._propagate_survey(XYZ_ref_start, E_ref_start,
                                                backtrack=False)
# drift by length
XYZ_temp, E_elem_temp = xt.survey.advance_element(XYZ_temp, E_elem_temp,
                          length=transf_params_start.shift_s)
# rotate by rot_start
XYZ_temp, E_elem_temp = rot_start._propagate_survey(XYZ_temp, E_elem_temp,
                                              backtrack=False)
XYZ_elem_start = XYZ_temp.copy()
E_elem_start = E_elem_temp.copy()

##### ref_end -> elem_end

trans_end = xt.Translation(
    shift_x=transf_params_end.shift_x,
    shift_y=transf_params_end.shift_y,
)
rot_end = xt.Rotation(
    rot_x_rad=-(transf_params_end.rot_x_rad),
    rot_y_rad=transf_params_end.rot_y_rad,
    rot_s_rad=transf_params_end.rot_s_rad_no_frame,
    seq='sxy'
)

# shift back by trans_end
XYZ_temp, E_elem_temp = trans_end._propagate_survey(XYZ_ref_end, E_ref_end,
                                                    backtrack=True)
# drift back by length
XYZ_temp, E_elem_temp = xt.survey.advance_element(XYZ_temp, E_elem_temp,
                          length=-transf_params_end.shift_s)
# rotate back by rot_end
XYZ_temp, E_elem_temp = rot_end._propagate_survey(XYZ_temp, E_elem_temp,
                                                  backtrack=True)

XYZ_elem_end = XYZ_temp.copy()
E_elem_end = E_elem_temp.copy()



plt.close('all')
plt.figure(1)
plt.plot(sv.Z, sv.X, '.-', label='survey')
plt.plot(XYZ_elem_start[2], XYZ_elem_start[0], 'o', label='start elem')
plt.plot(XYZ_elem_end[2], XYZ_elem_end[0], 'o', label='end elem')
plt.axis('equal')
plt.xlabel('Z [m]')
plt.ylabel('X [m]')
plt.legend()




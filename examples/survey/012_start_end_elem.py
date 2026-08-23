import xtrack as xt
import misalignment_entry_transforms as met
import numpy as np

env = xt.Environment()

env.new('q', 'Quadrupole', length=2, k1=0.1,
        rot_shift_anchor=1, rot_y_rad=np.deg2rad(30))
line = env.new_line(length=4, components=[
    env.place('q', at=2)
])

sv = line.survey()

name = 'q'
elem = env.get(name)

transf_params_start = met.get_entry_transform(
    shift_x=elem.shift_x,
    shift_y=elem.shift_y,
    shift_s=elem.shift_s,
    rot_y_rad=elem.rot_y_rad,
    rot_x_rad=elem.rot_x_rad,
    rot_s_rad_no_frame=elem.rot_s_rad_no_frame,
    rot_shift_anchor=elem.rot_shift_anchor,
    length=elem.length,
    angle=0.,
    h=0.0,
    rot_s_rad=elem.rot_s_rad)

transf_params_end = met.get_exit_transform(
    shift_x=elem.shift_x,
    shift_y=elem.shift_y,
    shift_s=elem.shift_s,
    rot_y_rad=elem.rot_y_rad,
    rot_x_rad=elem.rot_x_rad,
    rot_s_rad_no_frame=elem.rot_s_rad_no_frame,
    rot_shift_anchor=elem.rot_shift_anchor,
    length=elem.length,
    angle=0.,
    h=0.0,
    rot_s_rad=elem.rot_s_rad)


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



import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(sv.Z, sv.X, '.-', label='survey')
plt.plot(XYZ_elem_start[2], XYZ_elem_start[0], 'o', label='start elem')
plt.plot(XYZ_elem_end[2], XYZ_elem_end[0], 'o', label='end elem')
plt.axis('equal')
plt.xlabel('Z [m]')
plt.ylabel('X [m]')
plt.legend()






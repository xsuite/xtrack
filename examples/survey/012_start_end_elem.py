import xtrack as xt
import misalignment_entry_transforms as met

env = xt.Environment()

env.new('q', 'Quadrupole', length=2, k1=0.1,
        rot_shift_anchor=0.3, rot_y_rad=0.1)

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

rot_start = xt.Rotation(
    rot_x_rad=transf_params_start.rot_x_rad,
    rot_y_rad=transf_params_start.rot_y_rad,
    rot_s_rad=transf_params_start.rot_s_rad_no_frame,
)
trans_start = xt.Translation(
    shift_x=transf_params_start.shift_x,
    shift_y=transf_params_start.shift_y,
    shift_s=transf_params_start.shift_s,
)
rot_end = xt.Rotation(
    rot_x_rad=transf_params_end.rot_x_rad,
    rot_y_rad=transf_params_end.rot_y_rad,
    rot_s_rad=transf_params_end.rot_s_rad_no_frame,
)
trans_end = xt.Translation(
    shift_x=transf_params_end.shift_x,
    shift_y=transf_params_end.shift_y,
    shift_s=transf_params_end.shift_s,
)

XYZ_ref_start = sv['XYZ', name]
E_ref_start = sv['E_matrix', name]

XYZ_temp, E_elem_temp = rot_start._propagate_survey(XYZ_ref_start, E_ref_start,
                                              backtrack=False)
XYZ_temp, E_elem_temp = trans_start._propagate_survey(XYZ_temp, E_elem_temp,
                                                backtrack=False)

XYZ_elem_start = XYZ_temp.copy()
E_elem_start = E_elem_temp.copy()

assert hasattr(elem, 'isthick') and elem.isthick
XYZ_temp, E_elem_temp = xt.survey.advance_element(
    XYZ_temp, E_elem_temp,
    length=elem.length,
    angle=getattr(elem, 'angle', 0.),
    tilt=getattr(elem, 'rot_s_rad', 0.)
)
XYZ_elem_end = XYZ_temp.copy()
E_elem_end = E_elem_temp.copy()



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

XYZ_ref_start = sv['XYZ', name]
E_ref_start = sv['E_matrix', name]
elem = env.get(name)

transf_params = met.get_entry_transform(
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

rot = xt.Rotation(
    rot_x_rad=transf_params.rot_x_rad,
    rot_y_rad=transf_params.rot_y_rad,
    rot_s_rad=transf_params.rot_s_rad_no_frame,
)
trans = xt.Translation(
    shift_x=transf_params.shift_x,
    shift_y=transf_params.shift_y,
    shift_s=transf_params.shift_s,
)

XYZ_temp, E_elem_temp = rot._propagate_survey(XYZ_ref_start, E_ref_start,
                                              backtrack=False)
XYZ_temp, E_elem_temp = trans._propagate_survey(XYZ_temp, E_elem_temp,
                                                backtrack=False)



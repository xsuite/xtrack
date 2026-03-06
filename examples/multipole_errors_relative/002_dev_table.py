from pyparsing import line

import xtrack as xt
import numpy as np

# TODO:
# - Slices and replicas

env = xt.Environment()

line1 = env.new_line(components=[
    env.new('bend', xt.Bend, length=0.1, angle=0.01, at=0.3,
            knl=[0.002, 0.03, 0.4, 5, 6],
            ksl=[0.003, 0.04, 0.5, 6, 7],
            knl_rel=[0.1, 0.2, 0.3, 0.4, 0.5],
            ksl_rel=[0.1, 0.2, 0.3, 0.4, 0.5]),
    env.new('rbend', xt.RBend, length_straight=0.1, angle=0.01, at=0.6,
            knl=[0.003, 0.04, 0.5, 6, 7],
            ksl=[0.004, 0.05, 0.6, 7, 8],
            knl_rel=[0.2, 0.3, 0.4, 0.5, 0.6],
            ksl_rel=[0.7, 0.6, 0.5, 0.4, 0.3]),
    env.new('quad', xt.Quadrupole, length=0.1, k1=2, at=1,
            knl=[0.001, 0.02, 0.3, 4, 5],
            ksl=[0.002, 0.03, 0.4, 5, 6],
            knl_rel=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            ksl_rel=[0.8, 0.7, 0.6, 0.5, 0.4, 0.3]),
    env.new('skew_quad', xt.Quadrupole, length=0.1, k1s=2, main_is_skew=True, at=2,
            knl=[0.001, 0.02, 0.3, 4, 5],
            ksl=[0.002, 0.03, 0.4, 5, 6],
            knl_rel=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            ksl_rel=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
    env.new('sext', xt.Sextupole, length=0.1, k2=3, at=3,
            knl=[0.0005, 0.01, 0.2, 3, 4],
            ksl=[0.001, 0.02, 0.3, 4, 5],
            knl_rel=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ksl_rel=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
    env.new('skew_sext', xt.Sextupole, length=0.1, k2s=3, main_is_skew=True, at=4,
            knl=[0.0005, 0.01, 0.2, 3, 4],
            ksl=[0.001, 0.02, 0.3, 4, 5],
            knl_rel=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ksl_rel=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
    env.new('oct', xt.Octupole, length=0.1, k3=4, at=5),
    env.new('skew_oct', xt.Octupole, length=0.1, k3s=4, main_is_skew=True, at=6,
            knl=[0.0001, 0.001, 0.01, 2, 3],
            ksl=[0.0002, 0.002, 0.02, 3, 4],
            knl_rel=[0.5, 0.6, 0.7, 0.8, 0.9, 1],
            ksl_rel=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
    env.new('multipole', xt.Multipole, length=0.1, knl=[0,0,0,0,2],
            main_order=4, at=7,
            ksl=[1,2,3,4,5],
            knl_rel=[0.1, 0.2, 0.3, 0.4, 0.5],
            ksl_rel=[0.5, 0.4, 0.3, 0.2, 0.1],
            ),
    env.new('skew_multipole', xt.Multipole, length=0.1, ksl=[0,0,0,0,3],
            main_is_skew=True, main_order=4, at=8,
            knl=[5,4,3,2,1],
             knl_rel=[0.5, 0.4, 0.3, 0.2, 0.1],
             ksl_rel=[0.1, 0.2, 0.3, 0.4, 0.5],
             ),
])
line2 = line1.copy(shallow=True)
line2.slice_thick_elements(slicing_strategies=[
    xt.Strategy(slicing=xt.Teapot(1, mode='thick'))])

line = line1 + line2

attr = line.attr

if not line._has_valid_tracker():
    line.build_tracker()

main_order = attr['_own_main_order'] + attr['_parent_main_order']

mask_take_main_order = attr._cache['_own_main_order']._mask | attr._cache['_parent_main_order']._mask

_main_strength_normal = np.zeros(len(main_order), dtype=np.float64)
_main_strength_skew = np.zeros(len(main_order), dtype=np.float64)

element_type = line.tracker._tracker_data_base._line_table.element_type[:-1] # remove _end_point
parent_type = line.tracker._tracker_data_base._line_table.parent_type[:-1] # remove _end_point

MAX_ORDER = 5
for ii in range(MAX_ORDER+1):

    # Bends, RBends, Quadrupoles, and Sextupoles, Octupoles have implicit main order
    mask_type = None
    if ii == 0:
        mask_type = ((element_type == 'RBend') | (element_type == 'Bend')
                     | (parent_type == 'RBend') | (parent_type == 'Bend'))
    elif ii == 1:
        mask_type = ((element_type == 'Quadrupole') | (parent_type == 'Quadrupole'))
    elif ii == 2:
        mask_type = ((element_type == 'Sextupole') | (parent_type == 'Sextupole'))
    elif ii == 3:
        mask_type = ((element_type == 'Octupole') | (parent_type == 'Octupole'))

    this_norm = attr[f'_k{ii}l_no_rel']
    this_skew = attr[f'_k{ii}sl_no_rel']

    if mask_type is not None and np.any(mask_type):
        _main_strength_normal[mask_type] = this_norm[mask_type]
        _main_strength_skew[mask_type] = this_skew[mask_type]

    mask_main_order = (main_order == ii) & mask_take_main_order
    if np.any(mask_main_order):
        _main_strength_normal[mask_main_order] = this_norm[mask_main_order]
        _main_strength_skew[mask_main_order] = this_skew[mask_main_order]

main_is_skew = np.bool(attr['_own_main_is_skew'] + attr['_parent_main_is_skew'])

main_strength = np.zeros(len(main_order), dtype=np.float64)
main_strength[~main_is_skew] = _main_strength_normal[~main_is_skew]
main_strength[main_is_skew] = _main_strength_skew[main_is_skew]

tt = line.get_table()
tt['main_strength'] = main_strength
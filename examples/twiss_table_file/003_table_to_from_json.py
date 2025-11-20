import xtrack as xt

lhc= xt.load(['../../test_data/lhc_2024/lhc.seq',
              '../../test_data/lhc_2024/injection_optics.madx'],
                reverse_lines=['lhcb2'])
lhc.set_particle_ref('proton', energy0=450e9)

tw1 = lhc.lhcb1.twiss4d(coupling_edw_teng=True, strengths=True)

include = [
 'name',
 's',
#  'x',
#  'px',
#  'y',
#  'py',
#  'zeta',
#  'delta',
#  'ptau',
#  'W_matrix',
#  'kin_px',
#  'kin_py',
#  'kin_ps',
#  'kin_xprime',
#  'kin_yprime',
#  'name_env',
 'betx',
 'bety',
 'alfx',
 'alfy',
#  'gamx',
#  'gamy',
 'dx',
 'dpx',
 'dy',
 'dpy',
 'dx_zeta',
 'dpx_zeta',
 'dy_zeta',
 'dpy_zeta',
#  'betx1',
#  'bety1',
#  'betx2',
#  'bety2',
#  'alfx1',
#  'alfy1',
#  'alfx2',
#  'alfy2',
#  'gamx1',
#  'gamy1',
#  'gamx2',
#  'gamy2',
 'mux',
 'muy',
 'muzeta',
#  'nux',
#  'nuy',
#  'nuzeta',
#  'phix',
#  'phiy',
#  'phizeta',
#  'dmux',
#  'dmuy',
#  'dzeta',
 'bx_chrom',
 'by_chrom',
 'ax_chrom',
 'ay_chrom',
#  'wx_chrom',
#  'wy_chrom',
 'ddx',
 'ddpx',
 'ddy',
 'ddpy',
#  'c_minus_re',
#  'c_minus_im',
#  'c_r1',
#  'c_r2',
#  'c_phi1',
#  'c_phi2',
 'k0l',
 'k1l',
 'k2l',
#  'k3l',
#  'k4l',
#  'k5l',
 'k0sl',
 'k1sl',
 'k2sl',
#  'k3sl',
#  'k4sl',
#  'k5sl',
 'angle_rad',
 'rot_s_rad',
 'hkick',
 'vkick',
 'ks',
 'length',
 '_angle_force_body',
 'element_type',
#  'isthick',
#  'parent_name',
#  'r11_edw_teng',
#  'r12_edw_teng',
#  'r21_edw_teng',
#  'r22_edw_teng',
#  'betx_edw_teng',
#  'alfx_edw_teng',
#  'bety_edw_teng',
#  'alfy_edw_teng',
 'f1001',
 'f1010'
 ]

# Check json round trip
tw1.to_json('test.json')
twjson = xt.load('test.json')

# Check csv round trip
tw1.to_csv('test.csv')
twcsv = xt.load('test.csv')

# Check hdf5 round trip
tw1.to_hdf5('test.h5')
twh5 = xt.load('test.h5')

# Check tfs round trip
tw1.to_tfs('test.tfs', include=include)
twtfs = xt.TwissTable.from_tfs('test.tfs')



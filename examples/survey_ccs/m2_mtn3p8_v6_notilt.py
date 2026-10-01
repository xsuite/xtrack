"""
M2 beam line (version EYETS 2024-2025) defined with the xsuite environment API.

Line-by-line port of ``M2_MTN3p8_v6_notilt.seq`` (MAD-X sequence generated the
08-APR-2025 from https://layout.cern.ch). Conventions (same as ``xt.load``):

- all names are lower case (MAD-X is case insensitive);
- each MAD-X element class has a prototype element with the same name:
  COLLIMATOR / INSTRUMENT / MONITOR -> xt.Device, KICKER -> thick xt.Multipole
  (HKICK -> knl=[-hkick]), RBEND -> xt.RBend (L -> length_straight),
  TILT -> rot_s_rad, YROTATION -> xt.Rotation (ANGLE -> rot_y_rad),
  TRANSLATION -> xt.Translation (DX -> shift_x, DS -> shift_s);
- APERTYPE/APERTURE definitions are not ported (for now);
- ``slot_id`` is stored in ``element.extra``;
- variables not defined in the file (e.g. quadrupole and corrector strengths)
  are zero, as in MAD-X.
"""

import xtrack as xt

env = xt.Environment()
env.vars.default_to_zero = True  # undefined variables evaluate to zero (MAD-X behaviour)

# Prototypes for the MAD-X element classes (the information is kept in the
# `prototype` attribute of the elements)
env.new('collimator', xt.Device)
env.new('instrument', xt.Device)
env.new('monitor', xt.Device)
env.new('kicker', xt.Multipole, isthick=True)
env.new('marker', xt.Marker)
env.new('quadrupole', xt.Quadrupole)
env.new('rbend', xt.RBend)
env.new('yrotation', xt.Rotation)
env.new('translation', xt.Translation)


#==============================================================================
# TYPES DEFINITION
#==============================================================================


env['l.m2_bxsci'] = 0.1
env['l.m2_mbhhehwc'] = 2
env['l.m2_mbnh_hwp'] = 5
env['l.m2_mbnv_hwp'] = 5
env['l.m2_mbsmahwc'] = 1.1
env['l.m2_mbsmbhwc'] = 4
env['l.m2_mbxgdcwp'] = 3
env['l.m2_mbxhacwp'] = 2.5
env['l.m2_mcxcahwc'] = 0.4
env['l.m2_mcxcdhwc'] = 0.3
env['l.m2_mdpx'] = 0.3
env['l.m2_mqneetwc'] = 1
env['l.m2_mqnfbtwc'] = 2
env['l.m2_mtn__hwp'] = 3.6
env['l.m2_mbsa_hwp'] = 3.8
env['l.m2_omk'] = 0
env['l.m2_qnlb_8wp'] = 2.99
env['l.m2_qnrb_8wp'] = 2.99
env['l.m2_qwl__8wp'] = 2.948
env['l.m2_tbid'] = 0.25
env['l.m2_tcmab'] = 0.4
env['l.m2_xbms_001'] = 0.36
env['l.m2_xbms_002'] = 0.36
env['l.m2_xbms_003'] = 0.2
env['l.m2_xcbv'] = 1.2
env['l.m2_xchv_001'] = 1
env['l.m2_xciop001'] = 0.1
env['l.m2_xcmh'] = 5
env['l.m2_xcmib001'] = 1.6
env['l.m2_xcmib002'] = 3.2
env['l.m2_xcmv'] = 5
env['l.m2_xff__001'] = 0.276
env['l.m2_xffv'] = 0.276
env['l.m2_xffh'] = 0.276
env['l.m2_xion_001'] = 0.212
env['l.m2_xsci'] = 0.1
env['l.m2_xtax_009'] = 1.615
env['l.m2_xtax_010'] = 1.615
env['l.m2_xtcx_003'] = 2.4
env['l.m2_xvwaa001'] = 0.0001
env['l.m2_xvwad001'] = 0.0001
env['l.m2_xvwae001'] = 0.0002
env['l.m2_xvwaf001'] = 0.0001
env['l.m2_xvwai001'] = 0.0002
env['l.m2_xvwaj001'] = 0.0002
env['l.m2_xvwak001'] = 0.0002
env['l.m2_xvwmb001'] = 0.0002
env['l.m2_xvwmf001'] = 0.00012
env['l.m2_xvwmg001'] = 0.00012
env['l.m2_xvwam001'] = 0.00012
env['l.m2_xwcm_001'] = 0.1
env['l.m2_xwcm_002'] = 0.1
env['l.m2_xwcm_003'] = 0.1
env['l.m2_xcedn'] = 6.267  # Mechanical length
env['l.m2_vxss'] = 10.74422  # Vacuum chamber T6VXSS, length [m]
# ===== SCRAPER MULTIPOLE QUADRUPOLE STRENGTHS (knob-driven) =====
# Fit parameters for quadrupole term (order=2): can be changed for systematic/uncertainty studies

env['scra'] = 4.633689e-4*1e3  # amplitude [T/m]
env['scrl'] = 16.721577*1e-3  # decay length [m]
env['scrc'] = 4.065420e-5*1e3  # offset [T/m]
env['scrd'] = 1.085028e-7*1e6  # linear term [T/m^2]

env['beam_momentum'] = 160.0  # beam momentum [GeV/c]
env['scaling'] = '0.299792458/beam_momentum'  # scaling factor for conversion from T/m to kG/cm
# Each scraper quadrupole strength (as function of the actual aperture value in mm)
env['kquad.xcmh.x0610727_scr3h'] = '-1*scaling*(scra * exp(-kxcmh.x0610727_scr3h_coll1h / scrl) + scrc + scrd * kxcmh.x0610727_scr3h_coll1h)'
env['kquad.xcmh.x0610752_scr6h'] = '-1*scaling*(scra * exp(-kxcmh.x0610752_scr6h_coll2h / scrl) + scrc + scrd * kxcmh.x0610752_scr6h_coll2h)'
env['kquad.xcmh.x0610845_scr7h'] = '-1*scaling*(scra * exp(-kxcmh.x0610845_scr7h_coll3h / scrl) + scrc + scrd * kxcmh.x0610845_scr7h_coll3h)'

env['kquad.xcmv.x0610190_scr1v'] = '1*scaling*(scra * exp(-kxcmv.x0610190_scr1v_coll4v / scrl) + scrc + scrd * kxcmv.x0610190_scr1v_coll4v)'
env['kquad.xcmv.x0610715_scr2v'] = '1*scaling*(scra * exp(-kxcmv.x0610715_scr2v_coll5v / scrl) + scrc + scrd * kxcmv.x0610715_scr2v_coll5v)'
env['kquad.xcmv.x0610733_scr4v'] = '1*scaling*(scra * exp(-kxcmv.x0610733_scr4v_coll6v / scrl) + scrc + scrd * kxcmv.x0610733_scr4v_coll6v)'
env['kquad.xcmv.x0610741_scr5v'] = '1*scaling*(scra * exp(-kxcmv.x0610741_scr5v_coll7v / scrl) + scrc + scrd * kxcmv.x0610741_scr5v_coll7v)'
env['kquad.xcmv.x0610997_scr8v'] = '1*scaling*(scra * exp(-kxcmv.x0610997_scr8v_coll8v / scrl) + scrc + scrd * kxcmv.x0610997_scr8v_coll8v)'
env['kquad.xcmv.x0611050_scr9v'] = '1*scaling*(scra * exp(-kxcmv.x0611050_scr9v_coll9v / scrl) + scrc + scrd * kxcmv.x0611050_scr9v_coll9v)'

# ---------------------- COLLIMATOR     ---------------------------------------------
env.new('m2_tcmab', 'collimator', length='l.m2_tcmab')  # Collimation mask type B
env.new('m2_xcbv', 'collimator', length='l.m2_xcbv')  # Big vertical 2 blocks collimator
env.new('m2_xchv_001', 'collimator', length='l.m2_xchv_001')  # SPS Collimator horizontal et vertical 4 blocks (design 1970)
env.new('m2_xciop001', 'collimator', length='l.m2_xciop001')  # Converter IN OUT Plate - Lead 4 mm
env.new('m2_xcmh', 'quadrupole', length='l.m2_xcmh')  # Collimator Magnetic Horizontal
env.new('m2_xcmib001', 'collimator', length='l.m2_xcmib001')  # Magnetic Collimator Fixed (Magnetized Iron Block), 1.6 m
env.new('m2_xcmib002', 'collimator', length='l.m2_xcmib002')  # Magnetic Collimator Fixed (Magnetized Iron Block), 3.2 m
env.new('m2_xcmv', 'quadrupole', length='l.m2_xcmv')  # Collimator Magnetic Vertical
env.new('m2_xtax_009', 'collimator', length='l.m2_xtax_009')  # Target Absorber Type 009
env.new('m2_xtax_010', 'collimator', length='l.m2_xtax_010')  # Target Absorber Type 010
env.new('m2_xtcx_003', 'collimator', length='l.m2_xtcx_003')  # XTCX - Fixed collimator 2.4m, W inserts Ø40/120, Cooling, no Vacuum
# ---------------------- INSTRUMENT     ---------------------------------------------
env.new('m2_xvwaa001', 'instrument', length='l.m2_xvwaa001')  # X Vacuum Window, Aluminum [th=0.1], Tube DE 159, Flat Flange 192, aperture 120 + pumping port DN40, [L=100]
env.new('m2_xvwad001', 'instrument', length='l.m2_xvwad001')  # X Vacuum Window Aluminum [th=0.1], Tube DE 159, Flat Flange 192, aperture 120, [L=70] (VXW)
env.new('m2_xvwae001', 'instrument', length='l.m2_xvwae001')  # X Vacuum Window Aluminium EL900X60X0.2
env.new('m2_xvwaf001', 'instrument', length='l.m2_xvwaf001')  # X Vacuum Window Aluminium EL900X60X0.2 + pumping port
env.new('m2_xcedn', 'instrument', length='l.m2_xcedn')  # Cherenkov Differential Counter Nord
env.new('m2_xvwai001', 'instrument', length='l.m2_xvwai001')  # X Vacuum Window Aluminium [th=0.2], Flat Flanges 390, aperture EL350x60, [L=60] + pumping portDN40 (MTP)
env.new('m2_xvwaj001', 'instrument', length='l.m2_xvwaj001')  # X Vacuum Window Aluminum [th=0.2], Flat Al Flange DE 265, aperture 185 + pumping port, [L=110] (VDWP)
env.new('m2_xvwak001', 'instrument', length='l.m2_xvwak001')  # X Vacuum Window, Aluminum [th=0.2], Flat Al Flange DE 265, aperture 185, [L=70] (VDW)
env.new('m2_xvwmb001', 'instrument', length='l.m2_xvwmb001')  # X Vacuum Window, Mylar [th=0.25], Flat Al Flanges DE 265, aperture 185 + pumping port, [L=110] (VDWP)
env.new('m2_xvwmf001', 'instrument', length='l.m2_xvwmf001')  # X Vacuum Window, Mylar [th=0.125], Tube DE 159, Flat Flange 192, aperture 120 + pumping port DN40, [L=100]
env.new('m2_xvwmg001', 'instrument', length='l.m2_xvwmg001')  # X Vacuum Window Mylar [th=0.125] Tube DE 159, Flat Flange 192, aperture 120, [L=70] (VXW)
env.new('m2_xvwam001', 'instrument', length='l.m2_xvwam001')  # X Vacuum Window Mylar + pumping port DN40 DN120x100x0.12
# ---------------------- KICKER         ---------------------------------------------
env.new('m2_mcxcahwc', 'kicker', length='l.m2_mcxcahwc')  # Corrector magnet, H or V, type MDX
env.new('m2_mcxcdhwc', 'kicker', length='l.m2_mcxcdhwc')  # Corrector magnet, H or V, type MNPA30 - Old type name:  MNPA30.
env.new('m2_mdpx', 'kicker', length='l.m2_mdpx')  # Correcting dipole, H or V, type MDP, north area
# ---------------------- MARKER         ---------------------------------------------
env.new('m2_omk', 'marker')  # M2 markers (L := l.m2_omk dropped: marker has no length)
# ---------------------- MONITOR        ---------------------------------------------
env.new('m2_bxsci', 'monitor', length='l.m2_bxsci')  # Scintillator Counter Detector (Intensity Monitor)
env.new('m2_tbid', 'monitor', length='l.m2_tbid')  # target beam instrumentation, downstream
env.new('m2_xbms_001', 'monitor', length='l.m2_xbms_001')  # eXperimental Beam Momentum Station - Narrow and High (Long)
env.new('m2_xbms_002', 'monitor', length='l.m2_xbms_002')  # eXperimental Beam Momentum Station - Square (Long)
env.new('m2_xbms_003', 'monitor', length='l.m2_xbms_003')  # eXperimental Beam Momentum Station - Fibers (Short)
env.new('m2_xff__001', 'monitor', length='l.m2_xff__001')  # Filament Scintillator Profile Monitor
env.new('m2_xion_001', 'monitor', length='l.m2_xion_001')  # Assembly Ionization Chamber
env.new('m2_xsci', 'monitor', length='l.m2_xsci')  # Ensemble: Cadre et Scintillateur BXSCI
env.new('m2_xwcm_001', 'monitor', length='l.m2_xwcm_001')  # Ensemble: Multi Wire Proportional Chamber et Support cadre rouge not motorized
env.new('m2_xwcm_002', 'monitor', length='l.m2_xwcm_002')  # Ensemble: Multi Wire Proportional Chamber and Scintillator with Support cadre rouge not motorized
env.new('m2_xwcm_003', 'monitor', length='l.m2_xwcm_003')  # Ensemble: Multi Wire Proportional Chamber et Support cadre rouge motorized
env.new('m2_xffh', 'monitor', length='l.m2_xffh')  # Finger Scintillator Profile Monitor - Horizontal
env.new('m2_xffv', 'monitor', length='l.m2_xffv')  # Finger Scintillator Profile Monitor - Vertical
# ---------------------- QUADRUPOLE     ---------------------------------------------
env.new('m2_mqneetwc', 'quadrupole', length='l.m2_mqneetwc')  # Quadrupole magnet, type Q100, 1m - Magnetic length taken from EDMS 1786085
env.new('m2_mqnfbtwc', 'quadrupole', length='l.m2_mqnfbtwc')  # Quadrupole magnet, type Q200, 2m
env.new('m2_qnlb_8wp', 'quadrupole', length='l.m2_qnlb_8wp')  # Quadrupole, secondary beams, mineral isolation coil, north area
env.new('m2_qnrb_8wp', 'quadrupole', length='l.m2_qnrb_8wp')  # Quadrupole, secondary beams, reduced aperture, mineral isolation coil, north area
env.new('m2_qwl__8wp', 'quadrupole', length='l.m2_qwl__8wp')  # Quadrupole, Secondary Beams, West Area Type - dimensions according to drawing EDMS 350768
# ---------------------- RBEND          ---------------------------------------------
env.new('m2_mbhhehwc', 'rbend', length_straight='l.m2_mbhhehwc')  # Bending magnet, type M200, straight poles
env.new('m2_mbnh_hwp', 'rbend', length_straight='l.m2_mbnh_hwp')  # Bending magnet, secondary beams, horizontal, north area
env.new('m2_mbnv_hwp', 'rbend', length_straight='l.m2_mbnv_hwp')  # Bending magnet, secondary beams, vertical, north area - Mechanical dimensions from NORMA
env.new('m2_mbsmahwc', 'rbend', length_straight='l.m2_mbsmahwc')  # Experimental magnet Compass SM1
env.new('m2_mbsmbhwc', 'rbend', length_straight='l.m2_mbsmbhwc')  # Experimental magnet Compass SM2
env.new('m2_mbxgdcwp', 'rbend', length_straight='l.m2_mbxgdcwp')  # Bending Magnet, H or V, type MCW
env.new('m2_mbxhacwp', 'rbend', length_straight='l.m2_mbxhacwp')  # Bending Magnet, H or V, type VB1, 2.5m gap 108mm
env.new('m2_mtn__hwp', 'rbend', length_straight='l.m2_mtn__hwp')  # Bending magnet, Target N
env.new('m2_mbsa_hwp', 'rbend', length_straight='l.m2_mbsa_hwp')  # Bending magnet, Target N, longer MTN version
# ---------------------- CHAMBERS       ---------------------------------------------
env.new('m2_vxss', 'instrument', length='l.m2_vxss')  # Vacuum chamber T6VXSS, length 10.656 m
# ---------------------- MARKERS        ---------------------------------------------
env.new('m2_transform_marker', 'marker')  # Marker for transformation,
# ---------------------- INSTRUMENTS EXPERT NAMES          ---------------------------------------------
env.new('m2_abs', 'm2_xciop001')
env.new('m2_cedar1', 'm2_xcedn')
env.new('m2_cedar2', 'm2_xcedn')
# ---------------------- MONITOR EXPERT NAMES          ---------------------------------------------

env.new('m2_mwpc1_2', 'm2_xwcm_003')
env.new('m2_mwpc3_4', 'm2_xwcm_003')
env.new('m2_mwpc5_6', 'm2_xwcm_003')
env.new('m2_mwpc7_8', 'm2_xwcm_003')
env.new('m2_mwpc9_10', 'm2_xwcm_001')
env.new('m2_mwpc11_12', 'm2_xwcm_001')
env.new('m2_mwpc13_14', 'm2_xwcm_001')
env.new('m2_mwpc15_16', 'm2_xwcm_001')
env.new('m2_mwpc17_18', 'm2_xwcm_001')
env.new('m2_mwpc19_20', 'm2_xwcm_001')


#==============================================================================
# STRENGTH CONSTANTS
#==============================================================================

env['kmbh.x0610026_bend1h'] = 0.005953  # 0.0059333;
env['kmbh.x0610031_bend1h'] = 0.005953  # 0.0059333;
env['kmbh.x0610035_bend1h'] = 0.005953  # 0.0059333;
env['kmbh.x0610109_bend3h'] = 0.00793618  # 0.007939;
env['kmbh.x0611101_bend7h'] = -0.0
env['kmbh.x0611115_bend9h'] = -0.0
env['kmbv.x0610061_bend2v'] = -0.00317431  # -0.003172;
env['kmbv.x0610064_bend2v'] = -0.00317431  # -0.003172;
env['kmbv.x0610067_bend2v'] = -0.00317431  # -0.003172;
env['kmbv.x0610693_bend4v'] = -0.00479893  # -0.0048;
env['kmbv.x0610697_bend4v'] = -0.00479893  # -0.0048;
env['kmbv.x0610701_bend4v'] = -0.00479893  # -0.0048;
env['kmbv.x0610706_bend5v'] = -0.00960308  # -0.0096;
env['kmbv.x0611027_bend6v'] = 0.00999999  # 0.01;
env['kmbv.x0611033_bend6v'] = 0.00999999  # 0.01;
env['kmbv.x0611039_bend6v'] = 0.00999999  # 0.01;
env['kmbv.x0611111_bend8v'] = 0.00369

env['tilt.mbh.x0610026_bend1h'] = 0
env['tilt.mbh.x0610031_bend1h'] = 0
env['tilt.mbh.x0610035_bend1h'] = 0
env['tilt.mbv.x0610061_bend2v'] = 1.5707963267948966
env['tilt.mbv.x0610064_bend2v'] = 1.5707963267948966
env['tilt.mbv.x0610067_bend2v'] = 1.5707963267948966
env['tilt.mbh.x0610109_bend3h'] = 0
env['tilt.mbv.x0610693_bend4v'] = 1.5707963267948966
env['tilt.mbv.x0610697_bend4v'] = 1.5707963267948966
env['tilt.mbv.x0610701_bend4v'] = 1.5707963267948966
env['tilt.mbv.x0610706_bend5v'] = 1.5707963267948966
env['tilt.mbv.x0611027_bend6v'] = 1.5707963267948966
env['tilt.mbv.x0611033_bend6v'] = 1.5707963267948966
env['tilt.mbv.x0611039_bend6v'] = 1.5707963267948966
env['tilt.mbh.x0611101_bend7h'] = 0
env['tilt.mbv.x0611111_bend8v'] = 1.5707963267948966
env['tilt.mbh.x0611115_bend9h'] = 0
env['tilt.mbxh.x0611135_sm1'] = 0
env['tilt.mbxh.x0611149_sm2'] = 0
env['tilt.mcbh.x0610074_trim1h'] = 0
env['tilt.mcbv.x0610098_trim2v'] = 1.5707963267948966
env['tilt.mcbh.x0610650_trim3h'] = 0
env['tilt.mcbv.x0610651_trim4v'] = 1.5707963267948966
env['tilt.mcbh.x0610854_trim5h'] = 0
env['tilt.mcbh.x0610988_trim6h'] = 0
env['tilt.mcbv.x0610989_trim7v'] = 1.5707963267948966

#==============================================================================
# EXPERT NAMES
#==============================================================================
env['m2_version'] = 7

# Settings for M2_VERSION == 7
env['m2_angle1'] = 'kmbh.x0610026_bend1h*0.5'
env['m2_angle2'] = 'kmbh.x0610031_bend1h*0.5'
env['m2_angle3'] = 0
env['m2_rot_ang'] = -0.011906
env['m2_shift_hor'] = -0.21787286
env['m2_shift_long'] = 0.00150189


env.new('mbh.x0610026_bend1h', 'm2_mbsa_hwp', angle='m2_angle1', rot_s_rad='tilt.mbh.x0610026_bend1h', k0='2*sin(kmbh.x0610026_bend1h*0.5)/l.m2_mbsa_hwp')
env.new('mbh.x0610031_bend1h', 'm2_mbsa_hwp', angle='m2_angle2', rot_s_rad='tilt.mbh.x0610031_bend1h', k0='2*sin(kmbh.x0610031_bend1h*0.5)/l.m2_mbsa_hwp')
env.new('mbh.x0610035_bend1h', 'm2_mbsa_hwp', angle='m2_angle3', rot_s_rad='tilt.mbh.x0610035_bend1h', k0='2*sin(kmbh.x0610035_bend1h*0.5)/l.m2_mbsa_hwp')
env.new('mqd.x0610009_quad2d', 'm2_qnlb_8wp', k1='kmqd.x0610009_quad2d')
env.new('mqf.x0610012_quad3f', 'm2_qnlb_8wp', k1='kmqf.x0610012_quad3f')
env.new('mqf.x0610016_quad4f', 'm2_qnlb_8wp', k1='kmqf.x0610016_quad4f')
env.new('mqd.x0610019_quad5d', 'm2_qnlb_8wp', k1='kmqd.x0610019_quad5d')
env.new('mqd.x0610023_quad6d', 'm2_qnlb_8wp', k1='kmqd.x0610023_quad6d')
env.new('mqd.x0610005_quad1d', 'm2_qnrb_8wp', k1='kmqd.x0610005_quad1d')
env.new('xtax.x0610052_xtax1', 'm2_xtax_009')
env.new('xtax.x0610054_xtax2', 'm2_xtax_010')
env.new('xtcx.x0610002_xtcx1', 'm2_xtcx_003')
env.new('m2_rotation_in', 'yrotation', rot_y_rad='m2_rot_ang_in')
env.new('m2_rotation', 'yrotation', rot_y_rad='m2_rot_ang')
env.new('m2_shift_1', 'translation', shift_x='m2_shift_hor')
env.new('m2_shift_2', 'translation', shift_s='m2_shift_long')
env.new('mbh.x0611101_bend7h', 'm2_mbhhehwc', angle='kmbh.x0611101_bend7h', rot_s_rad='tilt.mbh.x0611101_bend7h')
env.new('mbv.x0611111_bend8v', 'm2_mbhhehwc', angle='kmbv.x0611111_bend8v', rot_s_rad='tilt.mbv.x0611111_bend8v')
env.new('mbh.x0611115_bend9h', 'm2_mbhhehwc', angle='kmbh.x0611115_bend9h', rot_s_rad='tilt.mbh.x0611115_bend9h')
env.new('mbh.x0610109_bend3h', 'm2_mbnh_hwp', angle='kmbh.x0610109_bend3h', rot_s_rad='tilt.mbh.x0610109_bend3h')
env.new('mbv.x0610706_bend5v', 'm2_mbnv_hwp', angle='kmbv.x0610706_bend5v', rot_s_rad='tilt.mbv.x0610706_bend5v')
env.new('mbv.x0611027_bend6v', 'm2_mbnv_hwp', angle='kmbv.x0611027_bend6v', rot_s_rad='tilt.mbv.x0611027_bend6v')
env.new('mbv.x0611033_bend6v', 'm2_mbnv_hwp', angle='kmbv.x0611033_bend6v', rot_s_rad='tilt.mbv.x0611033_bend6v')
env.new('mbv.x0611039_bend6v', 'm2_mbnv_hwp', angle='kmbv.x0611039_bend6v', rot_s_rad='tilt.mbv.x0611039_bend6v')
env.new('mbxh.x0611135_sm1', 'm2_mbsmahwc', rot_s_rad='tilt.mbxh.x0611135_sm1')
env.new('mbxh.x0611149_sm2', 'm2_mbsmbhwc', rot_s_rad='tilt.mbxh.x0611149_sm2')
env.new('mbv.x0610693_bend4v', 'm2_mbxgdcwp', angle='kmbv.x0610693_bend4v', rot_s_rad='tilt.mbv.x0610693_bend4v')
env.new('mbv.x0610697_bend4v', 'm2_mbxgdcwp', angle='kmbv.x0610697_bend4v', rot_s_rad='tilt.mbv.x0610697_bend4v')
env.new('mbv.x0610701_bend4v', 'm2_mbxgdcwp', angle='kmbv.x0610701_bend4v', rot_s_rad='tilt.mbv.x0610701_bend4v')
env.new('mbv.x0610061_bend2v', 'm2_mbxhacwp', angle='kmbv.x0610061_bend2v', rot_s_rad='tilt.mbv.x0610061_bend2v')
env.new('mbv.x0610064_bend2v', 'm2_mbxhacwp', angle='kmbv.x0610064_bend2v', rot_s_rad='tilt.mbv.x0610064_bend2v')
env.new('mbv.x0610067_bend2v', 'm2_mbxhacwp', angle='kmbv.x0610067_bend2v', rot_s_rad='tilt.mbv.x0610067_bend2v')
env.new('mcbh.x0610074_trim1h', 'm2_mcxcahwc', rot_s_rad='tilt.mcbh.x0610074_trim1h', knl=['-kmcbh.x0610074_trim1h'])
env.new('mcbv.x0610098_trim2v', 'm2_mcxcahwc', rot_s_rad='tilt.mcbv.x0610098_trim2v', knl=['-kmcbv.x0610098_trim2v'])
env.new('mcbv.x0610651_trim4v', 'm2_mcxcdhwc', rot_s_rad='tilt.mcbv.x0610651_trim4v', knl=['-kmcbv.x0610651_trim4v'])
env.new('mcbh.x0610650_trim3h', 'm2_mdpx', knl=['-kmcbh.x0610650_trim3h'])
env.new('mcbh.x0610854_trim5h', 'm2_mdpx', rot_s_rad='tilt.mcbh.x0610854_trim5h', knl=['-kmcbh.x0610854_trim5h'])
env.new('mcbh.x0610988_trim6h', 'm2_mdpx', rot_s_rad='tilt.mcbh.x0610988_trim6h', knl=['-kmcbh.x0610988_trim6h'])
env.new('mcbv.x0610989_trim7v', 'm2_mdpx', rot_s_rad='tilt.mcbv.x0610989_trim7v', knl=['-kmcbv.x0610989_trim7v'])
env.new('mqd.x0610112_quad12d', 'm2_mqneetwc', k1='kmqd.x0610112_quad12d')
env.new('mqf.x0610148_quad13f', 'm2_mqnfbtwc', k1='kmqf.x0610148_quad13f')
env.new('mqd.x0610184_quad14d', 'm2_mqnfbtwc', k1='kmqd.x0610184_quad14d')
env.new('mqf.x0610220_quad13f', 'm2_mqnfbtwc', k1='kmqf.x0610220_quad13f')
env.new('mqd.x0610256_quad14d', 'm2_mqnfbtwc', k1='kmqd.x0610256_quad14d')
env.new('mqf.x0610292_quad13f', 'm2_mqnfbtwc', k1='kmqf.x0610292_quad13f')
env.new('mqd.x0610328_quad14d', 'm2_mqnfbtwc', k1='kmqd.x0610328_quad14d')
env.new('mqf.x0610364_quad15f', 'm2_mqnfbtwc', k1='kmqf.x0610364_quad15f')
env.new('mqd.x0610400_quad16d', 'm2_mqnfbtwc', k1='kmqd.x0610400_quad16d')
env.new('mqf.x0610436_quad15f', 'm2_mqnfbtwc', k1='kmqf.x0610436_quad15f')
env.new('mqd.x0610472_quad16d', 'm2_mqnfbtwc', k1='kmqd.x0610472_quad16d')
env.new('mqf.x0610508_quad15f', 'm2_mqnfbtwc', k1='kmqf.x0610508_quad15f')
env.new('mqd.x0610544_quad16d', 'm2_mqnfbtwc', k1='kmqd.x0610544_quad16d')
env.new('mqf.x0610580_quad17f', 'm2_mqnfbtwc', k1='kmqf.x0610580_quad17f')
env.new('mqd.x0610616_quad18d', 'm2_mqnfbtwc', k1='kmqd.x0610616_quad18d')
env.new('mqf.x0610652_quad17f', 'm2_mqnfbtwc', k1='kmqf.x0610652_quad17f')
env.new('mqd.x0610654_quad19d', 'm2_mqnfbtwc', k1='kmqd.x0610654_quad19d')
env.new('mqd.x0610745_quad24d', 'm2_mqnfbtwc', k1='kmqd.x0610745_quad24d')
env.new('mqd.x0610748_quad25d', 'm2_mqnfbtwc', k1='kmqd.x0610748_quad25d')
env.new('mqf.x0610784_quad25f_inv', 'm2_mqnfbtwc', k1='kmqf.x0610784_quad25f_inv')
env.new('mqd.x0610820_quad25d', 'm2_mqnfbtwc', k1='kmqd.x0610820_quad25d')
env.new('mqf.x0610856_quad26f', 'm2_mqnfbtwc', k1='kmqf.x0610856_quad26f')
env.new('mqd.x0610900_quad27d', 'm2_mqnfbtwc', k1='kmqd.x0610900_quad27d')
env.new('mqf.x0610945_quad27f_inv', 'm2_mqnfbtwc', k1='kmqf.x0610945_quad27f_inv')
env.new('mqd.x0610990_quad27d', 'm2_mqnfbtwc', k1='kmqd.x0610990_quad27d')
env.new('mqd.x0610992_quad28d', 'm2_mqnfbtwc', k1='kmqd.x0610992_quad28d')
env.new('mqd.x0611072_quad33d', 'm2_mqnfbtwc', k1='kmqd.x0611072_quad33d')
env.new('mqd.x0611074_quad33d', 'm2_mqnfbtwc', k1='kmqd.x0611074_quad33d')
env.new('mqd.x0611077_quad33d', 'm2_mqnfbtwc', k1='kmqd.x0611077_quad33d')
env.new('mqd.x0611095_quad34d', 'm2_mqnfbtwc', k1='kmqd.x0611095_quad34d')
env.new('mqd.x0611097_quad34d', 'm2_mqnfbtwc', k1='kmqd.x0611097_quad34d')
env.new('amberta.x0611132', 'm2_omk')
env.new('mqf.x0610056_quad7f', 'm2_qwl__8wp', k1='kmqf.x0610056_quad7f')
env.new('mqf.x0610072_quad8f', 'm2_qwl__8wp', k1='kmqf.x0610072_quad8f')
env.new('mqd.x0610080_quad9d', 'm2_qwl__8wp', k1='kmqd.x0610080_quad9d')
env.new('mqd.x0610096_quad10d', 'm2_qwl__8wp', k1='kmqd.x0610096_quad10d')
env.new('mqf.x0610104_quad11f', 'm2_qwl__8wp', k1='kmqf.x0610104_quad11f')
env.new('mqf.x0610677_quad20f', 'm2_qwl__8wp', k1='kmqf.x0610677_quad20f')
env.new('mqd.x0610690_quad21d', 'm2_qwl__8wp', k1='kmqd.x0610690_quad21d')
env.new('mqd.x0610710_quad22d', 'm2_qwl__8wp', k1='kmqd.x0610710_quad22d')
env.new('mqf.x0610723_quad23f', 'm2_qwl__8wp', k1='kmqf.x0610723_quad23f')
env.new('mqf.x0611011_quad29f', 'm2_qwl__8wp', k1='kmqf.x0611011_quad29f')
env.new('mqd.x0611023_quad30d', 'm2_qwl__8wp', k1='kmqd.x0611023_quad30d')
env.new('mqd.x0611043_quad31d', 'm2_qwl__8wp', k1='kmqd.x0611043_quad31d')
env.new('mqf.x0611056_quad32f', 'm2_qwl__8wp', k1='kmqf.x0611056_quad32f')
env.new('mqf.x0611105_quad35f', 'm2_qwl__8wp', k1='kmqf.x0611105_quad35f')
env.new('mqf.x0611108_quad35f', 'm2_qwl__8wp', k1='kmqf.x0611108_quad35f')
env.new('mqd.x0611118_quad36d', 'm2_qwl__8wp', k1='kmqd.x0611118_quad36d')
env.new('mqd.x0611122_quad36d', 'm2_qwl__8wp', k1='kmqd.x0611122_quad36d')
env.new('bms_1', 'm2_xbms_001')
env.new('bms_4', 'm2_xbms_001')
env.new('bms_2', 'm2_xbms_002')
env.new('bms_3', 'm2_xbms_002')
env.new('bms_5', 'm2_xbms_003')
env.new('bms_6', 'm2_xbms_003')
env.new('xcbv.x0610858_coll5', 'm2_xcbv')
env.new('xchv.x0610058_coll1_2', 'm2_xchv_001')
env.new('xchv.x0610070_coll3_4', 'm2_xchv_001')
env.new('xchv.x0610288_coll10_11', 'm2_xchv_001')
env.new('xchv.x0611013_coll6_7', 'm2_xchv_001')
env.new('xchv.x0611054_coll8_9', 'm2_xchv_001')
env.new('xcmh.x0610727_scr3h', 'm2_xcmh', k1='kquad.xcmh.x0610727_scr3h')
env.new('xcmh.x0610752_scr6h', 'm2_xcmh', k1='kquad.xcmh.x0610752_scr6h')
env.new('xcmh.x0610845_scr7h', 'm2_xcmh', k1='kquad.xcmh.x0610845_scr7h')
env.new('xcm.x0610765_mib2', 'm2_xcmib001')
env.new('xcm.x0610767_mib2', 'm2_xcmib001')
env.new('xcm.x0611061_mib4', 'm2_xcmib001')
env.new('xcm.x0611063_mib4', 'm2_xcmib001')
env.new('xcm.x0611065_mib3', 'm2_xcmib001')
env.new('xcm.x0611066_mib3', 'm2_xcmib001')
env.new('xcm.x0611068_mib3', 'm2_xcmib001')
env.new('xcm.x0610226_mib1', 'm2_xcmib002')
env.new('xcm.x0610862_mib3', 'm2_xcmib002')
env.new('xcmv.x0610190_scr1v', 'm2_xcmv', k1='kquad.xcmv.x0610190_scr1v')
env.new('xcmv.x0610715_scr2v', 'm2_xcmv', k1='kquad.xcmv.x0610715_scr2v')
env.new('xcmv.x0610733_scr4v', 'm2_xcmv', k1='kquad.xcmv.x0610733_scr4v')
env.new('xcmv.x0610741_scr5v', 'm2_xcmv', k1='kquad.xcmv.x0610741_scr5v')
env.new('xcmv.x0610997_scr8v', 'm2_xcmv', k1='kquad.xcmv.x0610997_scr8v')
env.new('xcmv.x0611050_scr9v', 'm2_xcmv', k1='kquad.xcmv.x0611050_scr9v')
env.new('m2_fisc1v', 'm2_xff__001')
env.new('m2_fisc2h', 'm2_xff__001')
env.new('m2_fisc3v', 'm2_xff__001')
env.new('m2_fisc4h', 'm2_xff__001')

#==============================================================================
# SEQUENCE
#==============================================================================

line = env.new_line(name='m2', refer='centre', length=1185.6281, compose=True)
line.new('tbaca.x0600000', 'm2_omk', at=0, extra={'slot_id': 56964656})
line.new('tbid.251248', 'm2_tbid', at=.475, extra={'slot_id': 47601702})
line.new('tcmab.x0600001', 'm2_tcmab', at=.85, extra={'slot_id': 56992367})
line.new('xtcx.x0610002', 'xtcx.x0610002_xtcx1', at=2.4, extra={'slot_id': 56503593})
line.new('xvw.x0610003', 'm2_xvwaf001', at=3.69995, extra={'slot_id': 57603303})
line.new('qnrb.x0610005', 'mqd.x0610005_quad1d', at=5.465, extra={'slot_id': 56049083})
line.new('qnlb.x0610009', 'mqd.x0610009_quad2d', at=8.895, extra={'slot_id': 56500476})
line.new('qnlb.x0610012', 'mqf.x0610012_quad3f', at=12.325, extra={'slot_id': 56500489})
line.new('qnlb.x0610016', 'mqf.x0610016_quad4f', at=15.755, extra={'slot_id': 56500525})
line.new('qnlb.x0610019', 'mqd.x0610019_quad5d', at=19.185, extra={'slot_id': 56500534})
line.new('qnlb.x0610023', 'mqd.x0610023_quad6d', at=22.615, extra={'slot_id': 56500543})
line.new('tra.x0610026', 'm2_rotation_in', at=24.724997194472003, extra={'slot_id': 56500585})
line.new('mbsa.x0610026', 'mbh.x0610026_bend1h', at=26.625, extra={'slot_id': 56500585})
line.new('mbsa.x0610031', 'mbh.x0610031_bend1h', at=31.181, extra={'slot_id': 56500594})
line.new('mbsa.x0610035', 'mbh.x0610035_bend1h', at=35.737, extra={'slot_id': 56500603})
line.new('vxss.x0610045', 'm2_vxss', at=43.247, extra={'slot_id': 56500612})
line.new('ref.x0610046', 'm2_transform_marker', at=50.62)
line.new('tra.x0610046', 'm2_rotation', at=50.62)
line.new('shift.x0610046', 'm2_shift_1', at=50.62)
line.new('shift.x0610047', 'm2_shift_2', at=50.62)
line.new('xtax.x0610052', 'xtax.x0610052_xtax1', at=51.8775, extra={'slot_id': 56617856})
line.new('xtax.x0610054', 'xtax.x0610054_xtax2', at=53.5125, extra={'slot_id': 56617865})
line.new('xvw.x0610055', 'm2_xvwaf001', at=54.400965842, extra={'slot_id': 57603321})
line.new('qwl.x0610056', 'mqf.x0610056_quad7f', at=56.155, extra={'slot_id': 56500638})
line.new('xchv.x0610058', 'xchv.x0610058_coll1_2', at=58.475, extra={'slot_id': 56051764})
line.new('mbxha.x0610061', 'mbv.x0610061_bend2v', at=60.7, extra={'slot_id': 56048944})
line.new('mbxha.x0610064', 'mbv.x0610064_bend2v', at=64, extra={'slot_id': 56048953})
line.new('mbxha.x0610067', 'mbv.x0610067_bend2v', at=67.3, extra={'slot_id': 56048962})
line.new('xchv.x0610070', 'xchv.x0610070_coll3_4', at=69.525, extra={'slot_id': 56051773})
line.new('qwl.x0610072', 'mqf.x0610072_quad8f', at=71.845, extra={'slot_id': 56500647})
line.new('mcxca.x0610074', 'mcbh.x0610074_trim1h', at=74.10225, extra={'slot_id': 56049522})
line.new('qwl.x0610080', 'mqd.x0610080_quad9d', at=80.155, extra={'slot_id': 56500656})
line.new('qwl.x0610096', 'mqd.x0610096_quad10d', at=95.845, extra={'slot_id': 56500665})
line.new('mcxca.x0610098', 'mcbv.x0610098_trim2v', at=97.965, extra={'slot_id': 56049531})
line.new('xvw.x0610102', 'm2_xvwaa001', at=101.809968986, extra={'slot_id': 57603122})
line.new('xwcm.x0610102', 'm2_mwpc1_2', at=102.06, extra={'slot_id': 57718169})
line.new('xvw.x0610103', 'm2_xvwaa001', at=102.309968986, extra={'slot_id': 57603131})
line.new('qwl.x0610104', 'mqf.x0610104_quad11f', at=104.155, extra={'slot_id': 56500674})
line.new('mbnh.x0610109', 'mbh.x0610109_bend3h', at=108.67, extra={'slot_id': 56048971})
line.new('mqnee.x0610112', 'mqd.x0610112_quad12d', at=112.5, extra={'slot_id': 56049122})
line.new('mqnfb.x0610148', 'mqf.x0610148_quad13f', at=148, extra={'slot_id': 56049131})
line.new('mqnfb.x0610184', 'mqd.x0610184_quad14d', at=184, extra={'slot_id': 56049158})
line.new('xcmv.x0610190', 'xcmv.x0610190_scr1v', at=191, extra={'slot_id': 57015015})
line.new('xvw.x0610218', 'm2_xvwaf001', at=218.44995, extra={'slot_id': 57603330})
line.new('xwcm.x0610219', 'm2_mwpc3_4', at=218.51, extra={'slot_id': 57719098})
line.new('xvw.x0610219', 'm2_xvwaf001', at=218.749982319, extra={'slot_id': 57603339})
line.new('mqnfb.x0610220', 'mqf.x0610220_quad13f', at=220, extra={'slot_id': 56049140})
line.new('xcmib.x0610226', 'xcm.x0610226_mib1', at=226, extra={'slot_id': 56049034})
line.new('mqnfb.x0610256', 'mqd.x0610256_quad14d', at=256, extra={'slot_id': 56049167})
line.new('xchv.x0610288', 'xchv.x0610288_coll10_11', at=287.5)
line.new('mqnfb.x0610292', 'mqf.x0610292_quad13f', at=292, extra={'slot_id': 56049149})
line.new('mqnfb.x0610328', 'mqd.x0610328_quad14d', at=328, extra={'slot_id': 56049176})
line.new('mqnfb.x0610364', 'mqf.x0610364_quad15f', at=364, extra={'slot_id': 56049185})
line.new('mqnfb.x0610400', 'mqd.x0610400_quad16d', at=400, extra={'slot_id': 56049212})
line.new('mqnfb.x0610436', 'mqf.x0610436_quad15f', at=436, extra={'slot_id': 56049194})
line.new('mqnfb.x0610472', 'mqd.x0610472_quad16d', at=472, extra={'slot_id': 56049221})
line.new('mqnfb.x0610508', 'mqf.x0610508_quad15f', at=508, extra={'slot_id': 56049203})
line.new('xvw.x0610543', 'm2_xvwaa001', at=542.449982319, extra={'slot_id': 57603140})
line.new('xwcm.x0610543', 'm2_mwpc5_6', at=542.51, extra={'slot_id': 57719121})
line.new('xvw.x0610544', 'm2_xvwaf001', at=542.749982319, extra={'slot_id': 57603348})
line.new('mqnfb.x0610544', 'mqd.x0610544_quad16d', at=544, extra={'slot_id': 56049230})
line.new('mqnfb.x0610580', 'mqf.x0610580_quad17f', at=580, extra={'slot_id': 56049239})
line.new('mqnfb.x0610616', 'mqd.x0610616_quad18d', at=616, extra={'slot_id': 56049257})
line.new('xvw.x0610649', 'm2_xvwaa001', at=648.849982319, extra={'slot_id': 57603149})
line.new('xwcm.x0610649', 'm2_mwpc7_8', at=648.91, extra={'slot_id': 57719146})
line.new('xvw.x0610650', 'm2_xvwaa001', at=649.149982319, extra={'slot_id': 57603158})
line.new('mdpx.x0610650', 'mcbh.x0610650_trim3h', at=649.55, extra={'slot_id': 56500823})
line.new('mcxcd.x0610651', 'mcbv.x0610651_trim4v', at=650.35, extra={'slot_id': 56049540})
line.new('mqnfb.x0610652', 'mqf.x0610580_quad17f', at=652, extra={'slot_id': 56049248})
line.new('mqnfb.x0610654', 'mqd.x0610654_quad19d', at=654.5, extra={'slot_id': 56049266})
line.new('qwl.x0610677', 'mqf.x0610677_quad20f', at=677.461, extra={'slot_id': 56500683})
line.new('xion.x0610677', 'm2_xion_001', at=681.21, extra={'slot_id': 58596928})
line.new('xcio.x0610677', 'm2_xciop001', at=681.54, extra={'slot_id': 56503584})
line.new('xvw.x0610682', 'm2_xvwaa001', at=682, extra={'slot_id': 57603167})
line.new('xvw.x0610683', 'm2_xvwaa001', at=683, extra={'slot_id': 57603176})
line.new('qwl.x0610690', 'mqd.x0610690_quad21d', at=689.825, extra={'slot_id': 56500692})
line.new('xvw.x0610691', 'm2_xvwaa001', at=691.29905, extra={'slot_id': 57603185})
line.new('mbxgd.x0610693', 'mbv.x0610693_bend4v', at=693.36, extra={'slot_id': 56048980})
line.new('mbxgd.x0610697', 'mbv.x0610697_bend4v', at=697.06, extra={'slot_id': 56048989})
line.new('mbxgd.x0610701', 'mbv.x0610701_bend4v', at=700.76, extra={'slot_id': 56048998})
line.new('xwcm.x0610703', 'm2_mwpc9_10', at=702.66, extra={'slot_id': 57719169})
line.new('mbnv.x0610706', 'mbv.x0610706_bend5v', at=705.66, extra={'slot_id': 56500859})
line.new('qwl.x0610710', 'mqd.x0610710_quad22d', at=710.175, extra={'slot_id': 56500701})
line.new('xcmv.x0610715', 'xcmv.x0610715_scr2v', at=715.25, extra={'slot_id': 57015032})
line.new('xvw.x0610720', 'm2_xvwaa001', at=720, extra={'slot_id': 57603203})
line.new('qwl.x0610723', 'mqf.x0610723_quad23f', at=722.539, extra={'slot_id': 56500710})
line.new('xvw.x0610724', 'm2_xvwaa001', at=724.014, extra={'slot_id': 57603212})
line.new('xcmh.x0610727', 'xcmh.x0610727_scr3h', at=727.25, extra={'slot_id': 57014997})
line.new('xvw.x0610730', 'm2_xvwaa001', at=730, extra={'slot_id': 57603221})
line.new('xvw.x0610732', 'm2_xvwaa001', at=732.20, extra={'slot_id': 57603230})
line.new('xcmv.x0610733', 'xcmv.x0610733_scr4v', at=734.75, extra={'slot_id': 57015045})
line.new('xvw.x0610739', 'm2_xvwaa001', at=738.70, extra={'slot_id': 57603239})
line.new('xcmv.x0610741', 'xcmv.x0610741_scr5v', at=741.25, extra={'slot_id': 57015054})
line.new('xvw.x0610743', 'm2_xvwaf001', at=743.80, extra={'slot_id': 57603370})
line.new('mqnfb.x0610745', 'mqd.x0610745_quad24d', at=745.5, extra={'slot_id': 56617876})
line.new('mqnfb.x0610748', 'mqd.x0610748_quad25d', at=748, extra={'slot_id': 56049284})
line.new('xcmh.x0610752', 'xcmh.x0610752_scr6h', at=752.25, extra={'slot_id': 57015063})
line.new('xvw.x0610758', 'm2_xvwaf001', at=758, extra={'slot_id': 57603379})
line.new('xcmib.x0610765', 'xcm.x0610765_mib2', at=764.75, extra={'slot_id': 56049043})
line.new('xcmib.x0610767', 'xcm.x0610767_mib2', at=766.65, extra={'slot_id': 56049052})
line.new('mqnfb.x0610784', 'mqf.x0610784_quad25f_inv', at=784, extra={'slot_id': 56049293})
line.new('mqnfb.x0610820', 'mqd.x0610820_quad25d', at=820, extra={'slot_id': 56617885})
line.new('xcmh.x0610845', 'xcmh.x0610845_scr7h', at=842.7, extra={'slot_id': 57015086})
line.new('xvw.x0610843', 'm2_xvwam001', at=845.25, extra={'slot_id': 57603388})
line.new('xvw.x0610848', 'm2_xvwam001', at=848, extra={'slot_id': 57603397})
line.new('mdpx.x0610854', 'mcbh.x0610854_trim5h', at=854.35, extra={'slot_id': 56500832})
line.new('mqnfb.x0610856', 'mqf.x0610856_quad26f', at=856, extra={'slot_id': 56049311})
line.new('xvw.x0610858', 'm2_xvwam001', at=858, extra={'slot_id': 57603406})
line.new('xcbv.x0610858', 'xcbv.x0610858_coll5', at=859.04, extra={'slot_id': 56617897})
line.new('xcmib.x0610862', 'xcm.x0610862_mib3', at=862.2, extra={'slot_id': 57424624})
line.new('mqnfb.x0610900', 'mqd.x0610900_quad27d', at=900.8, extra={'slot_id': 56049320})
line.new('mqnfb.x0610945', 'mqf.x0610945_quad27f_inv', at=945.6, extra={'slot_id': 56049329})
line.new('mdpx.x0610988', 'mcbh.x0610988_trim6h', at=987.95, extra={'slot_id': 56500841})
line.new('mdpx.x0610989', 'mcbv.x0610989_trim7v', at=988.75, extra={'slot_id': 57430964})
line.new('mqnfb.x0610990', 'mqd.x0610990_quad27d', at=990.4, extra={'slot_id': 56049338})
line.new('mqnfb.x0610992', 'mqd.x0610992_quad28d', at=992.9, extra={'slot_id': 56049347})
line.new('xvw.x0610994', 'm2_xvwam001', at=994.15000016, extra={'slot_id': 57603446})
line.new('xcmv.x0610997', 'xcmv.x0610997_scr8v', at=997.91, extra={'slot_id': 57015123})
line.new('xvw.x0611001', 'm2_xvwam001', at=1000.61000016, extra={'slot_id': 57603455})
line.new('xwcm.x0611009', 'm2_mwpc11_12', at=1008.583, extra={'slot_id': 57719192})
line.new('xvw.x0611008', 'm2_xvwam001', at=1008.78300016, extra={'slot_id': 57603464})
line.new('xvw.x0611010', 'm2_xvwam001', at=1009.0729, extra={'slot_id': 57603473})
line.new('qwl.x0611011', 'mqf.x0611011_quad29f', at=1010.547, extra={'slot_id': 56500719})
line.new('xchv.x0611013', 'xchv.x0611013_coll6_7', at=1012.927, extra={'slot_id': 56051782})
line.new('qwl.x0611023', 'mqd.x0611023_quad30d', at=1022.911, extra={'slot_id': 56500728})
line.new('mbnv.x0611027', 'mbv.x0611027_bend6v', at=1027.426, extra={'slot_id': 56500868})
line.new('mbnv.x0611033', 'mbv.x0611033_bend6v', at=1033.086, extra={'slot_id': 56500878})
line.new('mbnv.x0611039', 'mbv.x0611039_bend6v', at=1038.746, extra={'slot_id': 56500887})
line.new('qwl.x0611043', 'mqd.x0611043_quad31d', at=1043.261, extra={'slot_id': 56500737})
line.new('xvw.x0611045', 'm2_xvwam001', at=1045, extra={'slot_id': 57603482})
line.new('xcmv.x0611050', 'xcmv.x0611050_scr9v', at=1049.651, extra={'slot_id': 57015166})
line.new('xchv.x0611054', 'xchv.x0611054_coll8_9', at=1053.245, extra={'slot_id': 56051791})
line.new('xvw.x0611055', 'm2_xvwam001', at=1054.1509, extra={'slot_id': 57603491})
line.new('qwl.x0611056', 'mqf.x0611056_quad32f', at=1055.625, extra={'slot_id': 56500746})
line.new('xvw.x0611057', 'm2_xvwam001', at=1057.41006266, extra={'slot_id': 57603500})
line.new('xwcm.x0611057', 'm2_mwpc13_14', at=1057.589, extra={'slot_id': 57719215})
line.new('xvw.x0611058', 'm2_xvwam001', at=1058.74906266, extra={'slot_id': 57603509})
line.new('xvw.x0611060', 'm2_xvwam001', at=1060, extra={'slot_id': 57603518})
line.new('xvw.x0611061', 'm2_xvwam001', at=1061, extra={'slot_id': 57603527})
line.new('xcmib.x0611061', 'xcm.x0611061_mib4', at=1062.159, extra={'slot_id': 57446660})
line.new('xcmib.x0611063', 'xcm.x0611063_mib4', at=1063.809, extra={'slot_id': 57446592})
line.new('xcmib.x0611065', 'xcm.x0611065_mib3', at=1065.459, extra={'slot_id': 57424687})
line.new('xcmib.x0611066', 'xcm.x0611066_mib3', at=1067.109, extra={'slot_id': 57446556})
line.new('xcmib.x0611068', 'xcm.x0611068_mib3', at=1068.759, extra={'slot_id': 57446520})
line.new('xvw.x0611069', 'm2_xvwam001', at=1069.96206266, extra={'slot_id': 57603536})
line.new('xvw.x0611070', 'm2_xvwam001', at=1070.82206266, extra={'slot_id': 57603545})
line.new('mqnfb.x0611072', 'mqd.x0611072_quad33d', at=1072.122, extra={'slot_id': 56049360})
line.new('mqnfb.x0611074', 'mqd.x0611074_quad33d', at=1074.622, extra={'slot_id': 56049369})
line.new('mqnfb.x0611077', 'mqd.x0611077_quad33d', at=1077.272, extra={'slot_id': 56049378})
line.new('xvw.x0611078', 'm2_xvwam001', at=1078.3, extra={'slot_id': 57603554})
line.new('xffv.x0611078', 'm2_fisc1v', at=1078.76, extra={'slot_id': 57718819})
line.new('xffh.x0611078', 'm2_fisc2h', at=1079.036, extra={'slot_id': 57718851})
line.new('xvw.x0611079', 'm2_xvwam001', at=1079.27406266, extra={'slot_id': 57603563})
line.new('xsci.x0611079', 'm2_xsci', at=1079.354, extra={'slot_id': 61178654})
line.new('xwcm.x0611079', 'm2_mwpc15_16', at=1079.514, extra={'slot_id': 57719249})
line.new('xced.x0611083', 'm2_cedar1', at=1082.8525, extra={'slot_id': 57572821})
line.new('xced.x0611089', 'm2_cedar2', at=1089.3695, extra={'slot_id': 57572958})
line.new('xvw.x0611091', 'm2_xvwam001', at=1092.53, extra={'slot_id': 57603572})
line.new('xsci.x0611092', 'm2_xsci', at=1092.678, extra={'slot_id': 61183374})
line.new('xion.x0611093', 'm2_xion_001', at=1093.238, extra={'slot_id': 61183549})
line.new('xvw.x0611093', 'm2_xvwam001', at=1093.39806266, extra={'slot_id': 57603581})
line.new('mqnfb.x0611095', 'mqd.x0611095_quad34d', at=1095.398, extra={'slot_id': 56049387})
line.new('mqnfb.x0611097', 'mqd.x0611097_quad34d', at=1097.898, extra={'slot_id': 56049401})
line.new('xffv.x0611099', 'm2_fisc3v', at=1099.414, extra={'slot_id': 57718933})
line.new('xffh.x0611099', 'm2_fisc4h', at=1099.69, extra={'slot_id': 57718956})
line.new('mbhhe.x0611101', 'mbh.x0611101_bend7h', at=1101.228, extra={'slot_id': 56049007})
line.new('xvw.x0611102', 'm2_xvwam001', at=1102.75706266, extra={'slot_id': 57603590})
line.new('xwcm.x0611102', 'm2_mwpc17_18', at=1102.837, extra={'slot_id': 57719272})
line.new('xvw.x0611103', 'm2_xvwam001', at=1102.91706266, extra={'slot_id': 57603599})
line.new('qwl.x0611105', 'mqf.x0611105_quad35f', at=1104.702, extra={'slot_id': 56500755})
line.new('qwl.x0611108', 'mqf.x0611108_quad35f', at=1108.132, extra={'slot_id': 56500764})
line.new('mbhhe.x0611111', 'mbv.x0611111_bend8v', at=1111.785, extra={'slot_id': 56049016})
line.new('xvw.x0611113', 'm2_xvwam001', at=1113.48506379, extra={'slot_id': 57603608})
line.new('xvw.x0611114', 'm2_xvwam001', at=1113.58506379, extra={'slot_id': 57603617})
line.new('mbhhe.x0611115', 'mbh.x0611115_bend9h', at=1115.285, extra={'slot_id': 56049025})
line.new('qwl.x0611118', 'mqd.x0611118_quad36d', at=1118.42, extra={'slot_id': 56500773})
line.new('qwl.x0611122', 'mqd.x0611122_quad36d', at=1121.85, extra={'slot_id': 56500782})
line.new('xwcm.x0611123', 'm2_mwpc19_20', at=1123.775, extra={'slot_id': 57719295})
line.new('xvw.x0611128', 'm2_xvwam001', at=1128, extra={'slot_id': 57603626})
line.new('exp.x0611132', 'amberta.x0611132', at=1131.824, extra={'slot_id': 62387268})
line.new('mbsma.x0611135', 'mbxh.x0611135_sm1', at=1135.324, extra={'slot_id': 56048926})
line.new('mbsmb.x0611149', 'mbxh.x0611149_sm2', at=1149.159, extra={'slot_id': 56048935})
line.new('xwcm.x0611185', 'm2_xwcm_002', at=1185.396, extra={'slot_id': 57829286})
line.new('xsci.x0611185', 'm2_bxsci', at=1185.5281, extra={'slot_id': 57829759})
line.end_compose()



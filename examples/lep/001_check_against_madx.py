import xtrack as xt

from cpymad.madx import Madx

mad = Madx()

#########################################
# Laod MAD-X model and import in Xsuite #
#########################################

mad = Madx()
mad.call('lep.seq9')
mad.call('lep.opt9')
mad.input('beam, particle=positron, energy=100, radiate=true;')
mad.use(sequence='lep')
mad.input('vrfc:=2.; vrfsc:=50.; vrfscn:=50.; ! LEP2  rf on')

line = xt.Line.from_madx_sequence(
    sequence=mad.sequence.LEP,
    allow_thick=True,
    enable_align_errors=True,
    deferred_expressions=True,
)
line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV,
                            gamma0=mad.sequence.LEP.beam.gamma)

#####################
# Slice Xsuite line #
#####################

line_thick = line.select() # Shallow copy
line.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(slicing=xt.Teapot(1)),
        xt.Strategy(slicing=xt.Teapot(4), element_type=xt.Bend),
        xt.Strategy(slicing=xt.Teapot(4), element_type=xt.Quadrupole),
    ])
line.to_json('lep_thin.json')

##################################
# Compute equilibrium emittances #
##################################

tw_norad = line.twiss()
line.configure_radiation('mean')
tw_rad = line.twiss(eneloss_and_damping=True)

##############################
# Compare against MAD-X emit #
##############################

mad.input('''
    beam, sequence=lep, particle=POSITRON, pc=100, radiate=true;
    use, sequence=lep;
    emit;
    ''')

mad_eneloss_turn = mad.table.emitsumm.u0[0]
mad_gemitt_x = mad.table.emitsumm.ex[0]
mad_gemitt_zeta = mad.table.emitsumm.et[0]
met = mad.table.emit.dframe()
mad_damping_constant_s_x = met[met.loc[:, 'parameter']=='damping_constant']['mode1'].values[0]
mad_damping_constant_s_y = met[met.loc[:, 'parameter']=='damping_constant']['mode2'].values[0]
mad_damping_constant_s_z = met[met.loc[:, 'parameter']=='damping_constant']['mode3'].values[0]
mad_partition_x = met[met.loc[:, 'parameter']=='damping_partion']['mode1'].values[0]
mad_partition_y = met[met.loc[:, 'parameter']=='damping_partion']['mode2'].values[0]
mad_partition_z = met[met.loc[:, 'parameter']=='damping_partion']['mode3'].values[0]

print(f'Energy loss per turn [eV]:')
print(f'- MAD-X:  {mad_eneloss_turn*1e9:.6g}')
print(f'- Xsuite: {tw_rad.eneloss_turn:.6g}')
print(f'Geometric emittance in x [m.rad]:')
print(f'- MAD-X:  {mad_gemitt_x:.6g}')
print(f'- Xsuite: {tw_rad.eq_gemitt_x:.6g}')
print(f'Geometric emittance in zeta [m.rad]:')
print(f'- MAD-X:  {mad_gemitt_zeta:.6g}')
print(f'- Xsuite: {tw_rad.eq_gemitt_zeta:.6g}')
print('Damping constants x [1/s]:')
print(f'- MAD-X:  {mad_damping_constant_s_x:.6g}')
print(f'- Xsuite: {tw_rad.damping_constants_s[0]:.6g}')
print('Damping constants y [1/s]:')
print(f'- MAD-X:  {mad_damping_constant_s_y:.6g}')
print(f'- Xsuite: {tw_rad.damping_constants_s[1]:.6g}')
print('Damping constants z [1/s]:')
print(f'- MAD-X:  {mad_damping_constant_s_z:.6g}')
print(f'- Xsuite: {tw_rad.damping_constants_s[2]:.6g}')

###########
# Asserts #
###########

import xobjects as xo
xo.assert_allclose(tw_rad['eneloss_turn'],  mad.table.emitsumm.u0*1e9, atol=0, rtol=1e-5)
xo.assert_allclose(tw_rad['eq_gemitt_x'],  mad.table.emitsumm.ex, atol=0, rtol=5e-3)
xo.assert_allclose(tw_rad['eq_gemitt_zeta'],  mad.table.emitsumm.et, atol=0, rtol=1e-3)

xo.assert_allclose(tw_rad['damping_constants_s'][0],
    mad_damping_constant_s_x,
    rtol=3e-3, atol=0
    )
xo.assert_allclose(tw_rad['damping_constants_s'][1],
    mad_damping_constant_s_y,
    rtol=1e-3, atol=0
    )
xo.assert_allclose(tw_rad['damping_constants_s'][2],
    mad_damping_constant_s_z,
    rtol=3e-3, atol=0
    )

xo.assert_allclose(tw_rad['partition_numbers'][0],
    mad_partition_x,
    rtol=3e-3, atol=0
    )
xo.assert_allclose(tw_rad['partition_numbers'][1],
    mad_partition_y,
    rtol=1e-3, atol=0
    )
xo.assert_allclose(tw_rad['partition_numbers'][2],
    mad_partition_z,
    rtol=3e-3, atol=0
    )

import xtrack as xt

from cpymad.madx import Madx

mad1=Madx()
mad1.call('../../test_data/hllhc15_thick/lhc.seq')
mad1.call('../../test_data/hllhc15_thick/hllhc_sequence.madx')
mad1.input('beam, sequence=lhcb1, particle=proton, energy=7000;')
mad1.use('lhcb1')
mad1.call("../../test_data/hllhc15_thick/opt_round_150_1500.madx")
mad1.twiss()

mad4=Madx()
mad4.input('mylhcbeam=4')
mad4.call('../../test_data/hllhc15_thick/lhcb4.seq')
mad4.call('../../test_data/hllhc15_thick/hllhc_sequence.madx')
mad4.input('beam, sequence=lhcb2, particle=proton, energy=7000;')
mad4.use('lhcb2')
mad4.call("../../test_data/hllhc15_thick/opt_round_150_1500.madx")
mad4.twiss()

line1=xt.Line.from_madx_sequence(mad1.sequence.lhcb1,
                                 allow_thick=True,
                                 deferred_expressions=True,
                                 replace_in_expr={'bv_aux':'bvaux_b1'})
line1.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7000e9)

line4=xt.Line.from_madx_sequence(mad4.sequence.lhcb2,
                                 allow_thick=True,
                                 deferred_expressions=True,
                                 replace_in_expr={'bv_aux':'bvaux_b2'})
line4.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7000e9)

# Remove solenoids (cannot backtwiss for now)
for ll in [line1, line4]:
    tt = ll.get_table()
    for nn in tt.rows[tt.element_type=='Solenoid'].name:
        ee_elen = ll[nn].length
        ll.element_dict[nn] = xt.Drift(length=ee_elen)

collider = xt.Environment(lines={'lhcb1':line1,'lhcb2':line4})
collider.lhcb1.particle_ref=xt.Particles(mass0=xt.PROTON_MASS_EV,p0c=7000e9)
collider.lhcb2.particle_ref=xt.Particles(mass0=xt.PROTON_MASS_EV,p0c=7000e9)

collider.lhcb1.twiss_default['method']='4d'
collider.lhcb2.twiss_default['method']='4d'
collider.lhcb2.twiss_default['reverse']=True

collider.build_trackers()

collider.to_json('collider_00_from_madx.json')

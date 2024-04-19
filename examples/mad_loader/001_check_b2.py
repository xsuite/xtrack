import xtrack as xt
import xpart as xp
import numpy as np

from cpymad.madx import Madx

mad1=Madx()
mad1.call('../../test_data/hllhc15_thick/lhc.seq')
mad1.call('../../test_data/hllhc15_thick/hllhc_sequence.madx')
mad1.input('beam, sequence=lhcb1, particle=proton, energy=7000;')
mad1.use('lhcb1')
mad1.input('beam, sequence=lhcb2, particle=proton, energy=7000, bv=-1;')
mad1.use('lhcb2')
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

for mad in [mad1, mad4]:
    mad.globals['vrf400'] = 16 # Check voltage expressions
    mad.globals['lagrf400.b2'] = 0.02 # Check lag expressions
    mad.globals['on_x1'] = 100 # Check kicker expressions
    mad.globals['on_sep2'] = 2 # Check kicker expressions
    mad.globals['on_x5'] = 123 # Check kicker expressions
    mad.globals['kqtf.b2'] = 1e-5 # Check quad expressions
    mad.globals['ksf.b2'] = 1e-3  # Check sext expressions
    mad.globals['kqs.l3b2'] = 1e-4 # Check skew expressions
    mad.globals['kss.a45b2'] = 1e-4 # Check skew sext expressions
    mad.globals['kof.a34b2'] = 3 # Check oct expressions
    mad.globals['on_crab1'] = -190 # Check cavity expressions
    mad.globals['on_crab5'] = -130 # Check cavity expressions
    mad.globals['on_sol_atlas'] = 1 # Check solenoid expressions
    mad.globals['kcdx3.r1'] = 1e-4 # Check thin decapole expressions
    mad.globals['kcdsx3.r1'] = 1e-4 # Check thin skew decapole expressions
    mad.globals['kctx3.l1'] = 1e-5 # Check thin dodecapole expressions
    mad.globals['kctsx3.r1'] = 1e-5 # Check thin skew dodecapole expressions

line2=xt.Line.from_madx_sequence(mad1.sequence.lhcb2,
                                 allow_thick=True,
                                 deferred_expressions=True,
                                 replace_in_expr={'bv_aux':'bvaux_b2'})
line2.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=7000e9)

line4=xt.Line.from_madx_sequence(mad4.sequence.lhcb2,
                                 allow_thick=True,
                                 deferred_expressions=True,
                                 replace_in_expr={'bv_aux':'bvaux_b2'})
line4.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=7000e9)

# Bend done

# Quadrupole
assert np.isclose(line2['mq.27l2.b2'].k1, line4['mq.27l2.b2'].k1, rtol=0, atol=1e-12)
assert np.isclose(line2['mqs.27l3.b2'].k1s, line4['mqs.27l3.b2'].k1s, rtol=0, atol=1e-12)

tt2 = line2.get_table()
tt4 = line4.get_table()

tt2nodr = tt2.rows[tt2.element_type != 'Drift']
tt4nodr = tt4.rows[tt4.element_type != 'Drift']

# Check s
l2names = list(tt2nodr.name)
l4names = list(tt4nodr.name)

l2names.remove('lhcb2$start')
l2names.remove('lhcb2$end')
l4names.remove('lhcb2$start')
l4names.remove('lhcb2$end')

# They seem not to be in the same place?!?
l2names.remove('tclia.4r2_entry')
l2names.remove('tclia.4r2_exit')
l4names.remove('tclia.4r2_entry')
l4names.remove('tclia.4r2_exit')

assert set(l2names) == set(l4names)

assert np.allclose(
    tt2nodr.rows[l2names].s, tt4nodr.rows[l2names].s, rtol=0, atol=1e-8)

for nn in l2names:
    print(nn+'              ', end='\r', flush=True)
    if nn == '_end_point':
        continue
    e2 = line2[nn]
    e4 = line4[nn]
    d2 = e2.to_dict()
    d4 = e4.to_dict()
    for kk in d2.keys():
        if kk == '__class__':
            assert d2[kk] == d4[kk]
            continue
        assert np.allclose(d2[kk], d4[kk], rtol=1e-10, atol=1e-16)
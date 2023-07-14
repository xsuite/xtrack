import xtrack as xt
import xpart as xp

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
line1.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=7000e9)

line4=xt.Line.from_madx_sequence(mad4.sequence.lhcb2,
                                 allow_thick=True,
                                 deferred_expressions=True,
                                 replace_in_expr={'bv_aux':'bvaux_b2'})
line4.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=7000e9)

collider = xt.Multiline(lines={'lhcb1':line1,'lhcb2':line4})
collider.lhcb1.particle_ref=xp.Particles(mass0=xp.PROTON_MASS_EV,p0c=7000e9)
collider.lhcb2.particle_ref=xp.Particles(mass0=xp.PROTON_MASS_EV,p0c=7000e9)

collider.lhcb1.twiss_default['method']='4d'
collider.lhcb2.twiss_default['method']='4d'
collider.lhcb2.twiss_default['reverse']=True

collider.build_trackers()

tw = collider.twiss()

nrj = 7000.
scale = 23348.89927*0.9
scmin = 0.03*7000./nrj
qtlimitx28 = 1.0*225.0/scale
qtlimitx15 = 1.0*205.0/scale
qtlimit2 = 1.0*160.0/scale
qtlimit3 = 1.0*200.0/scale
qtlimit4 = 1.0*125.0/scale
qtlimit5 = 1.0*120.0/scale
qtlimit6 = 1.0*90.0/scale


collider.vars.vary_default.update({
    'kqtf.a12b1': {'step': 1e-8, 'limits': None},
    'kqtf.a12b2': {'step': 1e-8, 'limits': None},
    'kqtd.a12b1': {'step': 1e-8, 'limits': None},
    'kqtd.a12b2': {'step': 1e-8, 'limits': None},
    'kqtf.a23b1': {'step': 1e-8, 'limits': None},
    'kqtf.a23b2': {'step': 1e-8, 'limits': None},
    'kqtd.a23b1': {'step': 1e-8, 'limits': None},
    'kqtd.a23b2': {'step': 1e-8, 'limits': None},
    'kqtf.a34b1': {'step': 1e-8, 'limits': None},
    'kqtf.a34b2': {'step': 1e-8, 'limits': None},
    'kqtd.a34b1': {'step': 1e-8, 'limits': None},
    'kqtd.a34b2': {'step': 1e-8, 'limits': None},
    'kqtf.a45b1': {'step': 1e-8, 'limits': None},
    'kqtf.a45b2': {'step': 1e-8, 'limits': None},
    'kqtd.a45b1': {'step': 1e-8, 'limits': None},
    'kqtd.a45b2': {'step': 1e-8, 'limits': None},
    'kqtf.a56b1': {'step': 1e-8, 'limits': None},
    'kqtf.a56b2': {'step': 1e-8, 'limits': None},
    'kqtd.a56b1': {'step': 1e-8, 'limits': None},
    'kqtd.a56b2': {'step': 1e-8, 'limits': None},
    'kqtf.a67b1': {'step': 1e-8, 'limits': None},
    'kqtf.a67b2': {'step': 1e-8, 'limits': None},
    'kqtd.a67b1': {'step': 1e-8, 'limits': None},
    'kqtd.a67b2': {'step': 1e-8, 'limits': None},
    'kqtf.a78b1': {'step': 1e-8, 'limits': None},
    'kqtf.a78b2': {'step': 1e-8, 'limits': None},
    'kqtd.a78b1': {'step': 1e-8, 'limits': None},
    'kqtd.a78b2': {'step': 1e-8, 'limits': None},
    'kqtf.a81b1': {'step': 1e-8, 'limits': None},
    'kqtf.a81b2': {'step': 1e-8, 'limits': None},
    'kqtd.a81b1': {'step': 1e-8, 'limits': None},
    'kqtd.a81b2': {'step': 1e-8, 'limits': None},
    'kqf.a12': {'step': 1e-10, 'limits': None},
    'kqd.a12': {'step': 1e-10, 'limits': None},
    'kqf.a23': {'step': 1e-10, 'limits': None},
    'kqd.a23': {'step': 1e-10, 'limits': None},
    'kqf.a34': {'step': 1e-10, 'limits': None},
    'kqd.a34': {'step': 1e-10, 'limits': None},
    'kqf.a45': {'step': 1e-10, 'limits': None},
    'kqd.a45': {'step': 1e-10, 'limits': None},
    'kqf.a56': {'step': 1e-10, 'limits': None},
    'kqd.a56': {'step': 1e-10, 'limits': None},
    'kqf.a67': {'step': 1e-10, 'limits': None},
    'kqd.a67': {'step': 1e-10, 'limits': None},
    'kqf.a78': {'step': 1e-10, 'limits': None},
    'kqd.a78': {'step': 1e-10, 'limits': None},
    'kqf.a81': {'step': 1e-10, 'limits': None},
    'kqd.a81': {'step': 1e-10, 'limits': None},

    'kq5.l2b1':    {'step': 1.0E-6, 'limits': ( qtlimit2*scmin, qtlimit2)},
    'kq4.l2b1':    {'step': 1.0E-6, 'limits': (-qtlimit2,       -qtlimit2*scmin)},
    'kq6.l2b1':    {'step': 1.0E-6, 'limits': (-qtlimit2,       -qtlimit2*scmin)},
    'kq7.l2b1':    {'step': 1.0E-6, 'limits': ( qtlimit3*scmin, qtlimit3)},
    'kq8.l2b1':    {'step': 1.0E-6, 'limits': (-qtlimit3,       -qtlimit3*scmin*0.9)},
    'kq9.l2b1':    {'step': 1.0E-6, 'limits': ( qtlimit3*scmin, qtlimit3)},
    'kq10.l2b':    {'step': 1.0E-6, 'limits': (-qtlimit3,       -qtlimit3*scmin)},
    'kqtl11.l2b1': {'step': 1.0E-6, 'limits': (-qtlimit4,       qtlimit4)},
    'kqt12.l2b1':  {'step': 1.0E-6, 'limits': (-qtlimit5,       qtlimit5)},
    'kqt13.l2b1':  {'step': 1.0E-6, 'limits': (-qtlimit5,       qtlimit5)},
    'kq4.r2b1':    {'step': 1.0E-6, 'limits': ( qtlimit2*scmin, qtlimit2)},
    'kq5.r2b1':    {'step': 1.0E-6, 'limits': (-qtlimit2,       -qtlimit2*scmin)},
    'kq6.r2b1':    {'step': 1.0E-6, 'limits': ( qtlimit2*scmin, qtlimit2)},
    'kq7.r2b1':    {'step': 1.0E-6, 'limits': (-qtlimit3,       -qtlimit3*scmin)},
    'kq8.r2b1':    {'step': 1.0E-6, 'limits': ( qtlimit3*scmin, qtlimit3)},
    'kq9.r2b1':    {'step': 1.0E-6, 'limits': (-qtlimit3,       -qtlimit3*scmin)},
    'kq10.r2b1':   {'step': 1.0E-6, 'limits': ( qtlimit3*scmin, qtlimit3)},
    'kqtl11.r2b1': {'step': 1.0E-6, 'limits': (-qtlimit4,       qtlimit4)},
    'kqt12.r2b1':  {'step': 1.0E-6, 'limits': (-qtlimit5,       qtlimit5)},
    'kqt13.r2b1':  {'step': 1.0E-6, 'limits': (-qtlimit5,       qtlimit5)},
})

collider.to_json('hllhc.json')
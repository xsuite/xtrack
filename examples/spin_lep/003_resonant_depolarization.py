import xtrack as xt
import xpart as xp
import xobjects as xo
import numpy as np

num_turns = 500

line = xt.Line.from_json('lep_sol.json')
line.particle_ref.anomalous_magnetic_moment=0.00115965218128
line.particle_ref.gamma0 = 89207.78287659843 # to have a spin tune of 103.45
spin_tune = line.particle_ref.anomalous_magnetic_moment[0]*line.particle_ref.gamma0[0]


# All off
line['on_sol.2'] = 0
line['on_sol.4'] = 0
line['on_sol.6'] = 0
line['on_sol.8'] = 0
line['on_spin_bump.2'] = 0
line['on_spin_bump.4'] = 0
line['on_spin_bump.6'] = 0
line['on_spin_bump.8'] = 0
line['on_coupl_sol.2'] = 0
line['on_coupl_sol.4'] = 0
line['on_coupl_sol.6'] = 0
line['on_coupl_sol.8'] = 0
line['on_coupl_sol_bump.2'] = 0
line['on_coupl_sol_bump.4'] = 0
line['on_coupl_sol_bump.6'] = 0
line['on_coupl_sol_bump.8'] = 0

# Match tunes
opt = line.match(
    method='4d',
    solve=False,
    vary=xt.VaryList(['kqf', 'kqd'], step=1e-4),
    targets=xt.TargetSet(qx=65.10, qy=71.20, tol=1e-4)
)
opt.solve()

tw = line.twiss4d(spin=True, radiation_integrals=True)

line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False

p0 = tw.particle_on_co.copy()
p0.spin_x = 1e-4
line.track(p0, num_turns=1000, turn_by_turn_monitor=True,
              with_progress=10)
mon0 = line.record_last_track

tt = line.get_table()
tt_bend = tt.rows[(tt.element_type == 'Bend') | (tt.element_type == 'RBend')]
tt_quad = tt.rows[(tt.element_type == 'Quadrupole')]

line.set(tt_bend, model='drift-kick-drift-expanded', integrator='uniform',
        num_multipole_kicks=3)

line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False

class Chirper(xt.BeamElement):

    _xofields = {
        'k0sl': xo.Float64,
        'q_start': xo.Float64,
        'q_end': xo.Float64,
        'num_turns': xo.Float64,
    }

    _extra_c_sources =['''
        /*gpufun*/
        void Chirper_track_local_particle(
                ChirperData el, LocalParticle* part0){

            double const k0sl = ChirperData_get_k0sl(el);
            double const q_start = ChirperData_get_q_start(el);
            double const q_end = ChirperData_get_q_end(el);
            double const num_turns = ChirperData_get_num_turns(el);

            //start_per_particle_block (part0->part)
                double const at_turn = LocalParticle_get_at_turn(part);
                double const qq = q_start + (q_end - q_start) * ((double) at_turn) / ((double) num_turns);
                double const dpy = k0sl * sin(2 * PI * qq * at_turn);
                LocalParticle_add_to_py(part, dpy);
            //end_per_particle_block
        }
        ''']

chirper = Chirper(
    k0sl=0,
    q_start=0,
    q_end=0,
    num_turns=0,
)
line.insert('chirper', obj=chirper, at='bfkv1.qs18.r2@start')

dq_excitation =  2e-4

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)

for iii in range(10):
    num_turns = 100000
    q_start_excitation = 0.448 + iii * dq_excitation
    # dq_excitation = 0.
    q_end_excitation = 0.449 + iii * dq_excitation
    k0sl_peak = 1e-6

    chirper.k0sl = k0sl_peak
    chirper.q_start = q_start_excitation
    chirper.q_end = q_end_excitation
    chirper.num_turns = num_turns

    p = tw.particle_on_co.copy()

    line.track(p, num_turns=num_turns, turn_by_turn_monitor=True,
            with_progress=1000)
    mon = line.record_last_track

    import nafflib
    spin_tune_freq_analysis = nafflib.get_tune(mon0.spin_x[0, :])

    freq_axis = np.linspace(q_start_excitation, q_end_excitation, num_turns)


    plt.plot(freq_axis, mon.spin_y.T)
    plt.xlabel('Excitation tune')
    plt.ylabel('Spin y')
    plt.axvline(spin_tune_freq_analysis)

    plt.show()
    plt.pause(1)


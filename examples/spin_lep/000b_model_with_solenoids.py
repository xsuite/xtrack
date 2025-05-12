import xtrack as xt

line = xt.Line.from_json('lep.json')

env = line.env


l_half_sol = 2.5
n_half_slices_sol = 10

env['ksol.2'] = 0
env['ksol.4'] = 0
env['ksol.6'] = 0
env['ksol.8'] = 0

for ipn in [2, 4, 6, 8]:
    for ii in range(n_half_slices_sol):
        env.new(
            f'sol_l_ip{ipn}..{ii}', xt.Solenoid, ks=f'ksol.{ipn}',
            length=l_half_sol/n_half_slices_sol)
        env.new(
            f'sol_r_ip{ipn}..{ii}', xt.Solenoid, ks=f'ksol.{ipn}',
            length=l_half_sol/n_half_slices_sol)

insertions = []
for ipn in [2, 4, 6, 8]:
    insertions += [
        env.place([f'sol_l_ip{ipn}..{ii}' for ii in range(n_half_slices_sol)],
                  anchor='end', at=-1e-12, from_=f'ip{ipn}'),
        env.place([f'sol_r_ip{ipn}..{ii}' for ii in range(n_half_slices_sol)],
                  anchor='start', at=1e-12, from_=f'ip{ipn}')]

line.insert(insertions)

line.vars.default_to_zero = True
line['ksol.2'] = '0.0079919339 * on_sol.2'
line['ksol.4'] = '0.0132567749 * on_sol.4'
line['ksol.6'] = '0.0034358215 * on_sol.6'
line['ksol.8'] = '0.0118735050 * on_sol.8'
line.vars.default_to_zero = False

# # March spin bump ip 2
line.vars.default_to_zero = True
line['kcv32.l2'] = '-3.14467e-05 * on_spin_bump.2'
line['kcv26.l2'] = '-6.28933e-05 * on_spin_bump.2'
line['kcv20.l2'] = '-3.14467e-05 * on_spin_bump.2'
line['kcv20.r2'] = '+3.14467e-05 * on_spin_bump.2'
line['kcv26.r2'] = '+6.28933e-05 * on_spin_bump.2'
line['kcv32.r2'] = '+3.14467e-05 * on_spin_bump.2'
line['kcv32.l4'] = '-5.21432e-05 * on_spin_bump.4'
line['kcv26.l4'] = '-0.000104286 * on_spin_bump.4'
line['kcv20.l4'] = '-5.21432e-05 * on_spin_bump.4'
line['kcv20.r4'] = '+5.21432e-05 * on_spin_bump.4'
line['kcv26.r4'] = '+0.000104286 * on_spin_bump.4'
line['kcv32.r4'] = '+5.21432e-05 * on_spin_bump.4'
line['kcv32.l6'] = '-1.35130e-05 * on_spin_bump.6'
line['kcv26.l6'] = '-2.70260e-05 * on_spin_bump.6'
line['kcv20.l6'] = '-1.35130e-05 * on_spin_bump.6'
line['kcv20.r6'] = '+1.35130e-05 * on_spin_bump.6'
line['kcv26.r6'] = '+2.70260e-05 * on_spin_bump.6'
line['kcv32.r6'] = '+1.35130e-05 * on_spin_bump.6'
line['kcv32.l8'] = '-4.67179e-05 * on_spin_bump.8'
line['kcv26.l8'] = '-9.34358e-05 * on_spin_bump.8'
line['kcv20.l8'] = '-4.67179e-05 * on_spin_bump.8'
line['kcv20.r8'] = '+4.67179e-05 * on_spin_bump.8'
line['kcv26.r8'] = '+9.34358e-05 * on_spin_bump.8'
line['kcv32.r8'] = '+4.67179e-05 * on_spin_bump.8'
line.vars.default_to_zero = False

line['on_sol.2'] = 0
line['on_sol.4'] = 0
line['on_sol.6'] = 0
line['on_sol.8'] = 0
line['on_spin_bump.2'] = 0
line['on_spin_bump.4'] = 0
line['on_spin_bump.6'] = 0
line['on_spin_bump.8'] = 0

tw = line.twiss4d(spin=True, radiation_integrals=True)

tw.plot('spin_z spin_x')

# Correct coupling ip2 sol alone
line['on_sol.2'] = 1

opt = line.match_knob(
    'on_coupl_sol.2',
    run=False,
    method='4d',
    vary=xt.VaryList(['kqt4.2', 'kqt3.2', 'kqt2.2', 'kqt1.l2']),
    targets=xt.TargetList(c_minus_re_0=0, c_minus_im_0=0, tol=1e-4),
)
opt.run_jacobian(40)
opt.generate_knob()

# Correct coupling ip2 sol and spin bump
line['on_sol.2'] = 1
line['on_spin_bump.2'] = 1
line['on_coupl_sol.2'] = 1
opt = line.match_knob(
    'on_coupl_sol_bump.2',
    run=False,
    method='4d',
    vary=xt.VaryList(['kqt4.2', 'kqt3.2', 'kqt2.2', 'kqt1.l2']),
    targets=xt.TargetList(c_minus_re_0=0, c_minus_im_0=0, tol=5e-4),
)
opt.run_jacobian(40)
opt.generate_knob()

line['on_sol.2'] = 0
line['on_spin_bump.2'] = 0
line['on_coupl_sol.2'] = 0
line['on_coupl_sol_bump.2'] = 0

# Correct coupling ip4 sol alone
line['on_sol.4'] = 1
opt = line.match_knob(
    'on_coupl_sol.4',
    run=False,
    method='4d',
    compute_chromatic_properties=False,
    vary=xt.VaryList(['kqt5.l4', 'kqt4.4', 'kqt3.4','kqt2.4',
                      'kqt1.l4', 'kqt1.r4', 'kqt5.r4'], step=1e-5),
    targets=xt.TargetList(c_minus_re_0=0, c_minus_im_0=0, tol=1e-4),
)
opt.run_jacobian(30)
opt.generate_knob()

# Correct coupling ip4 sol and spin bump
line['on_sol.4'] = 1
line['on_spin_bump.4'] = 1
line['on_coupl_sol.4'] = 1
opt = line.match_knob(
    'on_coupl_sol_bump.4',
    run=False,
    method='4d',
    compute_chromatic_properties=False,
    vary=xt.VaryList(['kqt5.l4', 'kqt4.4', 'kqt3.4','kqt2.4',
                      'kqt1.l4', 'kqt1.r4', 'kqt5.r4'], step=1e-5),
    targets=xt.TargetList(c_minus_re_0=0, c_minus_im_0=0, tol=5e-4),
)
opt.run_jacobian(30)
opt.generate_knob()

line['on_sol.4'] = 0
line['on_spin_bump.4'] = 0
line['on_coupl_sol.4'] = 0
line['on_coupl_sol_bump.4'] = 0

# Correct coupling ip6 sol alone
line['on_sol.6'] = 1
opt = line.match_knob(
    'on_coupl_sol.6',
    run=False,
    method='4d',
    compute_chromatic_properties=False,
    vary=xt.VaryList(['kqt4.6', 'kqt3.6', 'kqt2.6', 'kqt1.l6', 'kqt1.r6'],
                     step=1e-5),
    targets=xt.TargetList(c_minus_re_0=0, c_minus_im_0=0, tol=1e-4),
)
opt.run_jacobian(30)
opt.generate_knob()

# Correct coupling ip6 sol and spin bump
line['on_sol.6'] = 1
line['on_spin_bump.6'] = 1
line['on_coupl_sol.6'] = 1
opt = line.match_knob(
    'on_coupl_sol_bump.6',
    run=False,
    method='4d',
    compute_chromatic_properties=False,
    vary=xt.VaryList(['kqt4.6', 'kqt3.6', 'kqt2.6', 'kqt1.l6', 'kqt1.r6'],
                     step=1e-5),
    targets=xt.TargetList(c_minus_re_0=0, c_minus_im_0=0, tol=5e-4),
)
opt.run_jacobian(30)
opt.generate_knob()

line['on_sol.6'] = 0
line['on_spin_bump.6'] = 0
line['on_coupl_sol.6'] = 0
line['on_coupl_sol_bump.6'] = 0

# Correct coupling ip8 sol alone
line['on_sol.8'] = 1
opt = line.match_knob(
    'on_coupl_sol.8',
    run=False,
    method='4d',
    compute_chromatic_properties=False,
    vary=xt.VaryList(['kqt5.l8', 'kqt4.8', 'kqt3.8', 'kqt2.8', 'kqt1.l8',
                      'kqt1.r8', 'kqt5.r8'],
                     step=1e-5),
    targets=xt.TargetList(c_minus_re_0=0, c_minus_im_0=0, tol=1e-4),
)
opt.run_jacobian(30)
opt.generate_knob()

# Correct coupling ip8 sol and spin bump
line['on_sol.8'] = 1
line['on_spin_bump.8'] = 1
line['on_coupl_sol.8'] = 1
opt = line.match_knob(
    'on_coupl_sol_bump.8',
    run=False,
    method='4d',
    compute_chromatic_properties=False,
    vary=xt.VaryList(['kqt5.l8', 'kqt4.8', 'kqt3.8', 'kqt2.8', 'kqt1.l8',
                      'kqt1.r8', 'kqt5.r8'],
                     step=1e-5),
    targets=xt.TargetList(c_minus_re_0=0, c_minus_im_0=0, tol=5e-4),
)
opt.run_jacobian(30)
opt.generate_knob()

line['on_sol.8'] = 0
line['on_spin_bump.8'] = 0
line['on_coupl_sol.8'] = 0
line['on_coupl_sol_bump.8'] = 0

# All on
line['on_solenoids'] = 1
line['on_spin_bumps'] = 1
line['on_coupling_corrections'] = 1

# Set solenoids, spin bumps and coupling corrections
line['on_sol.2'] = 'on_solenoids'
line['on_sol.4'] = 'on_solenoids'
line['on_sol.6'] = 'on_solenoids'
line['on_sol.8'] = 'on_solenoids'
line['on_spin_bump.2'] = 'on_spin_bumps'
line['on_spin_bump.4'] = 'on_spin_bumps'
line['on_spin_bump.6'] = 'on_spin_bumps'
line['on_spin_bump.8'] = 'on_spin_bumps'
line['on_coupl_sol.2'] = 'on_coupling_corrections * on_solenoids'
line['on_coupl_sol.4'] = 'on_coupling_corrections * on_solenoids'
line['on_coupl_sol.6'] = 'on_coupling_corrections * on_solenoids'
line['on_coupl_sol.8'] = 'on_coupling_corrections * on_solenoids'
line['on_coupl_sol_bump.2'] = 'on_coupling_corrections * on_spin_bumps'
line['on_coupl_sol_bump.4'] = 'on_coupling_corrections * on_spin_bumps'
line['on_coupl_sol_bump.6'] = 'on_coupling_corrections * on_spin_bumps'
line['on_coupl_sol_bump.8'] = 'on_coupling_corrections * on_spin_bumps'

tt = line.get_table(attr=True)
tt_bend = tt.rows[(tt.element_type == 'RBend') | (tt.element_type == 'Bend')]
tt_quad = tt.rows[tt.element_type == 'Quadrupole']
tt_sext = tt.rows[tt.element_type == 'Sextupole']

# Set interators and multipole kicks
line.set(tt_bend, model='mat-kick-mat', integrator='uniform', num_multipole_kicks=5)
line.set(tt_quad, model='mat-kick-mat', integrator='uniform', num_multipole_kicks=5)

tw = line.twiss4d(spin=True, radiation_integrals=True)

line.to_json('lep_sol.json')

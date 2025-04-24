import xtrack as xt

line = xt.Line.from_json('lep.json')

env = line.env

env.new('half_sol', xt.Solenoid, length=2.5)

env['ksol.2'] = 0
env['ksol.4'] = 0
env['ksol.6'] = 0
env['ksol.8'] = 0

insertions = []
for ipn in [2, 4, 6, 8]:
    insertions += [
        env.new(
            f'sol_l_ip{ipn}', 'half_sol', ks=f'ksol.{ipn}',
            anchor='end', at=-1e-12, from_=f'ip{ipn}'), # PATCH!!!!
        env.new(
            f'sol_r_ip{ipn}', 'half_sol', ks=f'ksol.{ipn}',
            anchor='start', at=1e-12, from_=f'ip{ipn}'), # PATCH!!!!
        ]

line.insert(insertions)

line.vars.default_to_zero = True
line['ksol.2'] = '0.0079919339 * on_sol.2'
line['ksol.4'] = '0.0132567749 * on_sol.4'
line['ksol.6'] = '0.0034358215 * on_sol.6'
line['ksol.8'] = '0.0118735050 * on_sol.8'
line.vars.default_to_zero = False

# # March spin bump ip 2
line.vars.default_to_zero = True
line.vars['kcv32.l2'] = '-3.14467e-05 * on_spin_bump.2'
line.vars['kcv26.l2'] = '-6.28933e-05 * on_spin_bump.2'
line.vars['kcv20.l2'] = '-3.14467e-05 * on_spin_bump.2'
line.vars['kcv20.r2'] = '+3.14467e-05 * on_spin_bump.2'
line.vars['kcv26.r2'] = '+6.28933e-05 * on_spin_bump.2'
line.vars['kcv32.r2'] = '+3.14467e-05 * on_spin_bump.2'
line.vars['kcv32.l4'] = '-5.21432e-05 * on_spin_bump.4'
line.vars['kcv26.l4'] = '-0.000104286 * on_spin_bump.4'
line.vars['kcv20.l4'] = '-5.21432e-05 * on_spin_bump.4'
line.vars['kcv20.r4'] = '+5.21432e-05 * on_spin_bump.4'
line.vars['kcv26.r4'] = '+0.000104286 * on_spin_bump.4'
line.vars['kcv32.r4'] = '+5.21432e-05 * on_spin_bump.4'
line.vars['kcv32.l6'] = '-1.35130e-05 * on_spin_bump.6'
line.vars['kcv26.l6'] = '-2.70260e-05 * on_spin_bump.6'
line.vars['kcv20.l6'] = '-1.35130e-05 * on_spin_bump.6'
line.vars['kcv20.r6'] = '+1.35130e-05 * on_spin_bump.6'
line.vars['kcv26.r6'] = '+2.70260e-05 * on_spin_bump.6'
line.vars['kcv32.r6'] = '+1.35130e-05 * on_spin_bump.6'
line.vars['kcv32.l8'] = '-4.67179e-05 * on_spin_bump.8'
line.vars['kcv26.l8'] = '-9.34358e-05 * on_spin_bump.8'
line.vars['kcv20.l8'] = '-4.67179e-05 * on_spin_bump.8'
line.vars['kcv20.r8'] = '+4.67179e-05 * on_spin_bump.8'
line.vars['kcv26.r8'] = '+9.34358e-05 * on_spin_bump.8'
line.vars['kcv32.r8'] = '+4.67179e-05 * on_spin_bump.8'

line['on_sol.2'] = 1
line['on_sol.4'] = 0
line['on_sol.6'] = 1
line['on_sol.8'] = 1
line['on_spin_bump.2'] = 1
line['on_spin_bump.4'] = 0
line['on_spin_bump.6'] = 1
line['on_spin_bump.8'] = 1

tw = line.twiss4d(spin=True, radiation_integrals=True)

tw.plot('spin_z spin_x')



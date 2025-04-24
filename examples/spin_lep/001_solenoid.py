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
spin_tol = 1e-4
line['on_sol.2'] = 1
opt = line.match_knob(
    run=False,
    spin=True,
    default_tol=1e-10,
    knob_name='on_spin_corr_2.l',
    start = 'ip1', end='ip3',
    betx=1, bety=1, spin_y=1,
    vary=xt.VaryList(['kcv32.l2',
                      'kcv26.l2',
                      'kcv20.l2',
                      'kcv20.r2',
                      'kcv26.r2',
                      'kcv32.r2',
                      ], step=1e-6),
    targets=[
        xt.TargetSet(spin_x=xt.LessThan(spin_tol), at='ip2'),
        xt.TargetSet(spin_x=xt.GreaterThan(-spin_tol), at='ip2'),
        xt.TargetSet(spin_x=xt.LessThan(spin_tol), spin_z=xt.LessThan(spin_tol), at='sf.qf33.r2'),
        xt.TargetSet(spin_x=xt.GreaterThan(-spin_tol), spin_z=xt.GreaterThan(-spin_tol), at='sf.qf33.r2'),
        xt.TargetSet(y=0,    py=0, at='sd.qd20.l2', weight=100),
        xt.TargetSet(y=0,    py=0, at='sf.qf33.r2', weight=100),
    ])
opt.run_jacobian(10)






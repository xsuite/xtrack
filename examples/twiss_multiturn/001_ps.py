from cpymad.madx import Madx
import xtrack as xt

mad = Madx()
mad.input("""
beam, particle=proton, pc = 14.0;
BRHO      = BEAM->PC * 3.3356;
""")
mad.call("ps.seq")
mad.call("ps_hs_sftpro.str")
mad.use('ps')
twm = mad.twiss()

line = xt.Line.from_madx_sequence(mad.sequence.ps, allow_thick=True,
                                  deferred_expressions=True,
                                  replace_in_expr={'->':'__madarrow__'})
for kk in line.vars.keys():
    if '__madarrow__' in kk:
        mad_expr = kk.replace('__madarrow__', '->')
        mad.input(f'{kk} = {mad_expr}')
        line.vars[kk] = mad.globals[kk]

line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV,
                                    q0=1, gamma0=mad.sequence.ps.beam.gamma)
line.twiss_default['method'] = '4d'

tw = line.twiss()

opt = line.match(
    solve=False,
    vary=[
        xt.VaryList(['kf', 'kd'], step=1e-5),
    ],
    targets=[
        xt.TargetSet(qx=6.255278, qy=6.29826, tol=1e-7),
        ],
)
opt.solve()
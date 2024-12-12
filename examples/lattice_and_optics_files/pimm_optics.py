import xtrack as xt
env = xt.get_environment()

env.vars.update(default_to_zero=True,

    # Quadrupole strengths
    qf1k1 =  3.15396e-01,
    qd1k1 = -5.24626e-01,
    qf2k1 =  5.22717e-01,

    # Chromaticity sextupole strengths
    k2xcf = -4.33238e-01,
    k2xcd = -5.52276e-01,

    # Extraction sextupole strengths
    k2xrr_a = 'k2xrr',
    k2xrr =  8.65,
)

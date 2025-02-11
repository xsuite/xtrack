import xtrack as xt

env = xt.Environment()
env.vars.default_to_zero = True
env.new_line(components=[
    env.new('mq', 'Quadrupole', length=0.5, k1='kq'),
    env.new('mqs', 'Quadrupole', length=0.5, k1s='kqs'),
    env.new('mb', 'Bend', length=0.5, angle='ang', k0_from_h=True),
])


errors = {
    'mq': {'rel_knl': [1e-6, 1e-5, 1e-4], 'rel_ksl': [-1e-6, -1e-5, -1e-4]},
    'mqs': {'rel_knl': [2e-6, 2e-5, 2e-4], 'rel_ksl': [3e-6, 3e-5, 3e-4], 'refer': 'k1s'},
    'mb': {'rel_knl': [2e-6, 3e-5, 4e-4], 'rel_ksl': [5e-6, 6e-5, 7e-4]},
}

DEFAULT_REF_STRENGTH_NAME = {
    'Bend': 'k0',
    'Quadrupole': 'k1',
    'Sextupole': 'k2',
    'Octupole': 'k3',
}

for ele_name in errors:
    err = errors[ele_name]
    rel_knl = err.get('rel_knl', [])
    rel_ksl = err.get('rel_ksl', [])
    refer = err.get('refer', None)
    ele_class = env[ele_name].__class__.__name__
    if refer is not None:
        reference_strength_name = refer
    else:
        reference_strength_name = DEFAULT_REF_STRENGTH_NAME.get(ele_class, None)

    ref_str_ref = getattr(env.ref[ele_name], reference_strength_name)

    for ii, kk in enumerate(rel_knl):
        err_vname = f'err_{ele_name}_knl{ii}'
        env[err_vname] = kk
        if (env.ref[ele_name].knl[ii]._expr is None or env.ref[err_vname] in
                env.ref[ele_name].knl[ii]._expr._get_dependencies()):
            env[ele_name].knl[ii] += env.ref[err_vname] * ref_str_ref

    for ii, kk in enumerate(rel_ksl):
        err_vname = f'err_{ele_name}_ksl{ii}'
        env[err_vname] = kk
        if (env.ref[ele_name].ksl[ii]._expr is None or env.ref[err_vname] in
                env.ref[ele_name].ksl[ii]._expr._get_dependencies()):
            env[ele_name].ksl[ii] += env.ref[err_vname] * ref_str_ref

env['kq'] = 0.1
env['kqs'] = 0.2
env['ang'] = 0.3

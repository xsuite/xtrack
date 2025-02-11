import xtrack as xt

env = xt.Environment()
env.vars.default_to_zero = True
env.new_line(
    env.new('mq', 'Quadrupole', length=0.5, k1='kq'),
    env.new('mb', 'Bend', length=0.5, angle=0.1),
)

errors = {
    'mq': {'rel_knl': [1e-6, 1e-5, 1e-4], 'rel_ksl': [-1e-6, -1e-5, -1e-4]},
    'mb': {'rel_knl': [2e-6, 3e-5, 4e-4], 'rel_ksl': [5e-6, 6e-5, 7e-4]},
}

ele_name = 'mq'
err = errors[ele_name]
rel_knl = err.get('rel_knl', [])
rel_ksl = err.get('rel_ksl', [])

# TODO remember add/remove errors
# TODO need to handle relative errors

for ii, kk in enumerate(rel_knl):
    err_vname = f'err_{ele_name}_knl{ii}'
    env[err_vname] = kk
    k_ref = env.ref[ele_name].knl[ii]

    if (k_ref._expr is None or env.ref[err_vname] in
        env.ref[ele_name].knl[ii]._expr._get_dependencies()):

        env[ele_name].knl[ii] += env.ref[err_vname]

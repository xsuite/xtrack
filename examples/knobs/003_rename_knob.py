import xtrack as xt

env = xt.Environment()
env['aa'] = 5
env['bb'] = '2 * aa'

env.new('mb', 'Bend', length=1.0, angle='bb * 1e-3', knl=[0.0, '3*bb'])

assert str(env.ref['bb']._expr) == "(2.0 * vars['aa'])"
assert str(env.ref['mb'].angle._expr) == "(vars['bb'] * 0.001)"
assert str(env.ref['mb'].knl[1]._expr) == "(3.0 * vars['bb'])"

assert str(env.ref_manager.tasks) == "{vars['bb']: vars['bb'] = (2.0 * vars['aa']), element_refs['mb'].angle: element_refs['mb'].angle = (vars['bb'] * 0.001), element_refs['mb'].knl[1]: element_refs['mb'].knl[1] = (3.0 * vars['bb'])}"
assert str(env.ref_manager.rdeps) == "defaultdict(<class 'xdeps.refs.RefCount'>, {vars['aa']: {vars['bb']: 1}, vars['bb']: {element_refs['mb'].angle: 1, element_refs['mb']: 2, element_refs['mb'].knl[1]: 1, element_refs['mb'].knl: 1}})"
assert str(env.ref_manager.rtasks) == "defaultdict(<class 'xdeps.refs.RefCount'>, {vars['bb']: {element_refs['mb'].angle: 1, element_refs['mb'].knl[1]: 1}})"
assert str(env.ref_manager.deptasks) == "defaultdict(<class 'xdeps.refs.RefCount'>, {vars['t_turn_s']: {}, vars['aa']: {vars['bb']: 1}, vars['bb']: {element_refs['mb'].angle: 1, element_refs['mb'].knl[1]: 1}, element_refs['mb'].angle: {}, element_refs['mb']: {}, element_refs['mb'].knl[1]: {}, element_refs['mb'].knl: {}})"
assert str(env.ref_manager.tartasks) == "defaultdict(<class 'xdeps.refs.RefCount'>, {vars['aa']: {}, vars['bb']: {vars['bb']: 1}, element_refs['mb'].angle: {element_refs['mb'].angle: 1}, element_refs['mb']: {element_refs['mb'].angle: 1, element_refs['mb'].knl[1]: 1}, element_refs['mb'].knl[1]: {element_refs['mb'].knl[1]: 1}, element_refs['mb'].knl: {element_refs['mb'].knl[1]: 1}})"

env.vars.rename('bb', 'cc', verbose=True)

assert str(env.ref['cc']._expr) == "(2.0 * vars['aa'])"
assert str(env.ref['mb'].angle._expr) == "(vars['cc'] * 0.001)"
assert str(env.ref['mb'].knl[1]._expr) == "(3.0 * vars['cc'])"

assert str(env.ref_manager.tasks) == "{vars['cc']: vars['cc'] = (2.0 * vars['aa']), element_refs['mb'].angle: element_refs['mb'].angle = (vars['cc'] * 0.001), element_refs['mb'].knl[1]: element_refs['mb'].knl[1] = (3.0 * vars['cc'])}"
assert str(env.ref_manager.rdeps) == "defaultdict(<class 'xdeps.refs.RefCount'>, {vars['aa']: {vars['cc']: 1}, vars['cc']: {element_refs['mb'].angle: 1, element_refs['mb']: 2, element_refs['mb'].knl[1]: 1, element_refs['mb'].knl: 1}})"
assert str(env.ref_manager.rtasks) == "defaultdict(<class 'xdeps.refs.RefCount'>, {vars['cc']: {element_refs['mb'].angle: 1, element_refs['mb'].knl[1]: 1}})"
assert str(env.ref_manager.deptasks) == "defaultdict(<class 'xdeps.refs.RefCount'>, {vars['t_turn_s']: {}, vars['aa']: {vars['cc']: 1}, vars['cc']: {element_refs['mb'].angle: 1, element_refs['mb'].knl[1]: 1}, element_refs['mb'].angle: {}, element_refs['mb']: {}, element_refs['mb'].knl[1]: {}, element_refs['mb'].knl: {}})"
assert str(env.ref_manager.tartasks) == "defaultdict(<class 'xdeps.refs.RefCount'>, {vars['aa']: {}, vars['cc']: {vars['cc']: 1}, element_refs['mb'].angle: {element_refs['mb'].angle: 1}, element_refs['mb']: {element_refs['mb'].angle: 1, element_refs['mb'].knl[1]: 1}, element_refs['mb'].knl[1]: {element_refs['mb'].knl[1]: 1}, element_refs['mb'].knl: {element_refs['mb'].knl[1]: 1}})"

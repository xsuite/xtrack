import xtrack as xt

env = xt.Environment()
env['a'] = 10
env['b'] = 20
env.new('q1', 'Quadrupole', length='2.0 + b', knl=[0, '3*a'])
env.new('q2', 'q1', length='2.0 + b', knl=[0, '3*a'])

assert 'q1' in env.element_dict
assert len(env.ref['a']._find_dependant_targets()) == 7
# is:
# [vars['a'],
#  element_refs['q2'].knl[1],
#  element_refs['q2'],
#  element_refs['q2'].knl,
#  element_refs['q1'],
#  element_refs['q1'].knl[1],
#  element_refs['q1'].knl]

env._remove_element('q1')
assert 'q1' not in env.element_dict
assert len(env.ref['a']._find_dependant_targets()) == 4
# is:
# [vars['a'],
#  element_refs['q2'].knl[1],
#  element_refs['q2'],
#  element_refs['q2'].knl]

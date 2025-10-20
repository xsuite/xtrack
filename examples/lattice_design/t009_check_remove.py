import xtrack as xt
import pytest

env = xt.Environment()

env.vars['a1'] = 2.
env.vars['a2'] = 3.
env.vars['a3'] = 4.
env.vars['a4'] = 5.
env.elements['e1'] = xt.Quadrupole(length=1.)
env.elements['e2'] = xt.Quadrupole(length=2.)
env.elements['e3'] = xt.Quadrupole(length=3.)
env.elements['e4'] = xt.Quadrupole(length=4.)
env.particles['p1'] = xt.Particles(p0c=1e12)
env.particles['p2'] = xt.Particles(p0c=2e12)
env.particles['p3'] = xt.Particles(p0c=3e12)
env.particles['p4'] = xt.Particles(p0c=4e12)
env.lines['l1'] = env.new_line(length=3)
env.lines['l2'] = env.new_line(length=4)
env.lines['l3'] = env.new_line(length=5)
env.lines['l4'] = env.new_line(length=6)

with pytest.raises(KeyError):
    env.vars.remove('zz')

with pytest.raises(KeyError):
    env.elements.remove('zz')

with pytest.raises(KeyError):
    env.particles.remove('zz')

with pytest.raises(KeyError):
    env.lines.remove('zz')

with pytest.raises(KeyError):
    env.remove('zz')

assert 'a1' in env.vars
assert 'a1' in env
env.vars.remove('a1')
assert 'a1' not in env.vars
assert 'a1' not in env

assert 'a2' in env.vars
assert 'a2' in env
env.remove('a2')
assert 'a2' not in env.vars
assert 'a2' not in env

assert 'a3' in env.vars
assert 'a3' in env
del env.vars['a3']
assert 'a3' not in env.vars
assert 'a3' not in env

assert 'a4' in env.vars
assert 'a4' in env
del env['a4']
assert 'a4' not in env.vars
assert 'a4' not in env

assert 'e1' in env.elements
assert 'e1' in env
env.elements.remove('e1')
assert 'e1' not in env.elements
assert 'e1' not in env

assert 'e2' in env.elements
assert 'e2' in env
env.remove('e2')
assert 'e2' not in env.elements
assert 'e2' not in env

assert 'e3' in env.elements
assert 'e3' in env
del env.elements['e3']
assert 'e3' not in env.elements
assert 'e3' not in env

assert 'e4' in env.elements
assert 'e4' in env
del env['e4']
assert 'e4' not in env.elements
assert 'e4' not in env

assert 'p1' in env.particles
assert 'p1' in env
env.particles.remove('p1')
assert 'p1' not in env.particles
assert 'p1' not in env

assert 'p2' in env.particles
assert 'p2' in env
env.remove('p2')
assert 'p2' not in env.particles
assert 'p2' not in env

assert 'p3' in env.particles
assert 'p3' in env
del env.particles['p3']
assert 'p3' not in env.particles
assert 'p3' not in env

assert 'p4' in env.particles
assert 'p4' in env
del env['p4']
assert 'p4' not in env.particles
assert 'p4' not in env

assert 'l1' in env.lines
assert 'l1' in env
env.lines.remove('l1')
assert 'l1' not in env.lines
assert 'l1' not in env

assert 'l2' in env.lines
assert 'l2' in env
env.remove('l2')
assert 'l2' not in env.lines
assert 'l2' not in env

assert 'l3' in env.lines
assert 'l3' in env
del env.lines['l3']
assert 'l3' not in env.lines
assert 'l3' not in env

assert 'l4' in env.lines
assert 'l4' in env
del env['l4']
assert 'l4' not in env.lines
assert 'l4' not in env

env['a'] = 5.
env.new('e', 'Quadrupole', length='3*a')
with pytest.raises(RuntimeError):
    env.remove('a') # a is used by element e
env.remove('e')
env.remove('a')
assert 'e' not in env.elements
assert 'a' not in env.vars

env.new('e', 'Quadrupole', length=2)
env['a'] = 3 * env.ref['e'].length
assert env['a'] == 6.
assert str(env.ref['a']._expr) == "(3 * element_refs['e'].length)"
with pytest.raises(RuntimeError):
    env.remove('e') # e is used by variable a
env.remove('a')
env.remove('e')

env['a'] = 4.
env.new_particle('p', p0c='2*a*1e12')
assert env['p'].p0c == 8e12
with pytest.raises(RuntimeError):
    env.remove('a') # a is used by particle p
env.remove('p')
env.remove('a')

env.new_particle('p', p0c=1e12)
env['a'] = env.ref['p'].p0c
assert env['a'] == 1e12
assert str(env.ref['a']._expr) == "particles['p'].p0c"
with pytest.raises(RuntimeError):
    env.remove('p') # p is used by variable a
env.remove('a')
env.remove('p')

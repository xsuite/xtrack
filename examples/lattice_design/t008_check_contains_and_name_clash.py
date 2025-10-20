import xtrack as xt
import pytest

env = xt.Environment()

env.vars['a'] = 2.
env.elements['e1'] = xt.Quadrupole(length=1.)
env.particles['p1'] = xt.Particles(p0c=1e12)
env.lines['l1'] = env.new_line(length=3)

assert 'a' in env
assert 'a' in env.vars
assert 'a' not in env.elements
assert 'a' not in env.particles
assert 'a' not in env.lines

assert 'e1' in env
assert 'e1' in env.elements
assert 'e1' not in env.vars
assert 'e1' not in env.particles
assert 'e1' not in env.lines

assert 'p1' in env
assert 'p1' in env.particles
assert 'p1' not in env.vars
assert 'p1' not in env.elements
assert 'p1' not in env.lines

assert 'l1' in env
assert 'l1' in env.lines
assert 'l1' not in env.vars
assert 'l1' not in env.elements
assert 'l1' not in env.particles

assert 'zz' not in env
assert 'zz' not in env.vars
assert 'zz' not in env.elements
assert 'zz' not in env.particles
assert 'zz' not in env.lines

with pytest.raises(KeyError):
    _ = env['zz']

with pytest.raises(KeyError):
    _ = env.vars['zz']

with pytest.raises(KeyError):
    _ = env.elements['zz']

with pytest.raises(KeyError):
    _ = env.particles['zz']

with pytest.raises(KeyError):
    _ = env.lines['zz']

# ----- Check behavior of vars container -----

# Updating the variable should be possible
env.vars['a'] = 3.
assert env['a'] == 3.

with pytest.raises(ValueError):
    env.vars['e1'] = 5.  # Clash with element name

with pytest.raises(ValueError):
    env.vars['p1'] = 5.  # Clash with particle name

with pytest.raises(ValueError):
    env.vars['l1'] = 5.  # Clash with line name

with pytest.raises(ValueError):
    env.elements['a'] = xt.Marker()

# ----- Check behavior of elements container -----

with pytest.raises(ValueError):
    env.elements['a'] = xt.Marker()  # Clash with var name

with pytest.raises(ValueError):
    env.elements['e1'] = xt.Marker()  # Clash with existing element name

with pytest.raises(ValueError):
    env.elements['p1'] = xt.Marker()  # Clash with particle name

with pytest.raises(ValueError):
    env.elements['l1'] = xt.Marker()  # Clash with line name

# ----- Check behavior of particles container -----
with pytest.raises(ValueError):
    env.particles['a'] = xt.Particles()  # Clash with var name

with pytest.raises(ValueError):
    env.particles['e1'] = xt.Particles()  # Clash with element name

with pytest.raises(ValueError):
    env.particles['p1'] = xt.Particles()  # Clash with existing particle name

with pytest.raises(ValueError):
    env.particles['l1'] = xt.Particles()  # Clash with line name

# ----- Check behavior of lines container -----
with pytest.raises(ValueError):
    env.lines['a'] = env.new_line()  # Clash with var name

with pytest.raises(ValueError):
    env.lines['e1'] = env.new_line()  # Clash with element name

with pytest.raises(ValueError):
    env.lines['p1'] = env.new_line()  # Clash with particle name

with pytest.raises(ValueError):
    env.lines['l1'] = env.new_line()  # Clash with existing line name

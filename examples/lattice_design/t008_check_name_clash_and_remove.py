import xtrack as xt
import pytest

env = xt.Environment()

env.vars['a'] = 2.
env.elements['e1'] = xt.Quadrupole(length=1.)
env.particles['p1'] = xt.Particles(p0c=1e12)
env.lines['l1'] = env.new_line(length=3)

# Updating the variable should be possible
env.vars['a'] = 3.
assert env['a'] == 3.

with pytest.raises(ValueError):
    env.vars['e1'] = 5.  # Clash with element name

with pytest.raises(ValueError):
    env.vars['p1'] = 5.  # Clash with particle name

with pytest.raises(ValueError):
    env.vars['l1'] = 5.  # Clash with line name
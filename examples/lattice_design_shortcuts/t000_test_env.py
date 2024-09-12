import xtrack as xt
import xobjects as xo

env = xt.Environment()

env.vars({
    'k.1': 1.,
    'a': 2.,
    'b': '2 * a + k.1',
})

assert env.vv['b'] == 2 * 2 + 1

env.vars['a'] = env.vars['k.1']
assert env.vv['b'] == 2 * 1 + 1

env.vars(a=3.)
env.vars({'k.1': 'a'})
assert env.vv['k.1'] == 3.
assert env.vv['b'] == 2 * 3 + 3.

# line = env.new_line([
#     env.new
# ]
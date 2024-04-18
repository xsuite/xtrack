import numpy as np
import xtrack as xt

bend = xt.Bend(k0=0.4, h=0.3, length=1)

elements = {
    # correct case
    'a0': bend,
    'a1': xt.Replica(parent_name='a0'),
    'a2': xt.Replica(parent_name='a1'),
    # create a simple loop
    'b0': xt.Replica(parent_name='b0'),
    # loop with two replicas
    'c0': xt.Replica(parent_name='c1'),
    'c1': xt.Replica(parent_name='c0'),
    # bigger loop
    'd0': xt.Replica(parent_name='d1'),
    'd1': xt.Replica(parent_name='d2'),
    'd2': xt.Replica(parent_name='d0'),
    'd3': xt.Replica(parent_name='d1'),
}

line = xt.Line(elements=elements, element_names=list(elements.keys()))

ok = ['a1', 'a2']
for kk in ok:
    assert line[kk].resolve(line) is line['a0']

error = ['b0', 'c0', 'c1', 'd0', 'd1', 'd2', 'd3']
for kk in error:
    try:
        line[kk].resolve(line)
    except RecursionError:
        pass
    else:
        raise Exception(f'Element {kk} should not be resolvable')

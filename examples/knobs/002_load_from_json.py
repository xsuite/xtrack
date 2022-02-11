import json
import math

import xtrack as xt
import xdeps as xd

with open('status.json', 'r') as fid:
    dct = json.load(fid)

elements = dct['elements']
variables = dct['variables']

line = xt.Line.from_dict(dct['line'])
line.element_dict = dict(zip(line.element_names, line.element_list))

manager=xd.Manager()
vref=manager.ref(variables,'v')
eref=manager.ref(elements,'e')
fref=manager.ref(math,'f')
lref = manager.ref(line.element_dict, 'line_dict')
manager.reload(dct['xdeps_status'])

line.vars = vref

tracker = xt.Tracker(line=line)
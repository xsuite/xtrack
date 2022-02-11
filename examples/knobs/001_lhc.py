import json
import sys
import math
import numpy as np

import xtrack as xt
import xpart as xp
import xobjects as xo
import xdeps as xd
from xdeps.madxutils import MadxEval

from cpymad.madx import Madx
mad = Madx(command_log="mad_final.log")
mad.call('../../test_data/hllhc15_noerrors_nobb/sequence.madx')
mad.use(sequence="lhcb1")
mad.globals['vrf400'] = 16
mad.globals['lagrf400.b1'] = 0.5
mad.twiss()

# Extract all values
from collections import defaultdict

variables=defaultdict(lambda :0)
for name,par in mad.globals.cmdpar.items():
    variables[name]=par.value

elements={}
for name,elem in mad.elements.items():
    elemdata={}
    for parname, par in elem.cmdpar.items():
        elemdata[parname]=par.value
    elements[name]=elemdata
    elements[name]['__basetype__'] = elem.base_type.name

# Build expressions
manager=xd.Manager()
vref=manager.ref(variables,'v')
eref=manager.ref(elements,'e')
fref=manager.ref(math,'f')
madeval=MadxEval(vref,fref,eref).eval

for name,par in mad.globals.cmdpar.items():
    if par.expr is not None:
        vref[name]=madeval(par.expr)

for name,elem in mad.elements.items():
    for parname, par in elem.cmdpar.items():
        if par.expr is not None:
            if par.dtype==12: # handle lists
                for ii,ee in enumerate(par.expr):
                    if ee is not None:
                        eref[name][parname][ii]=madeval(ee)
            else:
                eref[name][parname]=madeval(par.expr)

line = xt.Line.from_madx_sequence(
        mad.sequence['lhcb1'], apply_madx_errors=False)
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1,
                        gamma0=mad.sequence.lhcb1.beam.gamma)
line.element_dict = dict(zip(line.element_names, line.element_list))

tracker = xt.Tracker(line=line)

lref = manager.ref(line.element_dict, 'line_dict')

for nn, ee in line.element_dict.items():
    if isinstance(ee, xt.Multipole):
        assert nn in elements.keys()
        ref_knl = line.element_dict[nn].knl.copy()
        ref_ksl = line.element_dict[nn].ksl.copy()
        if elements[nn]['__basetype__'] == 'hkicker':
            lref[nn].knl[0] = -eref[nn]['kick']
        elif elements[nn]['__basetype__'] == 'vkicker':
            lref[nn].ksl[0] = eref[nn]['kick']
        elif elements[nn]['__basetype__'] == 'multipole':
            lref[nn].knl = eref[nn]['knl']
            lref[nn].ksl[0] = eref[nn]['ksl'][0]
        elif elements[nn]['__basetype__'] in ['tkicker', 'kicker']:
            if hasattr(elements[nn], 'hkick'):
                lref[nn].knl[0] = -eref[nn]['hkick']
            if hasattr(elements[nn], 'vkick'):
                lref[nn].ksl[0] = eref[nn]['vkick']
        else:
            raise ValueError('???')
        assert np.allclose(line.element_dict[nn].knl, ref_knl, 1e-18)
        assert np.allclose(line.element_dict[nn].ksl, ref_ksl, 1e-18)

line.vars = vref

with open('status.json', 'w') as fid:
    json.dump({
            'line': line.to_dict(),
            'xdeps_status': manager.dump(),
            'variables': variables,
            'elements': elements,
        }, fid, cls=xo.JEncoder)
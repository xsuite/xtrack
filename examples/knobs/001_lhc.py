import sys
import math
import numpy as np

import xtrack as xt
import xpart as xp
import xobjects as xo
import xdeps as xd

from cpymad.madx import Madx

path = '../../test_data/hllhc14_input_mad/'

mad = Madx(command_log="mad_final.log")
mad.call(path + "final_seq.madx")
mad.use(sequence="lhcb1")
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

# Build expressions
manager=xd.Manager()
vref=manager.ref(variables,'v')
eref=manager.ref(elements,'e')
fref=manager.ref(math,'f')
madeval=xd.MadxEval(vref,fref,eref).eval

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
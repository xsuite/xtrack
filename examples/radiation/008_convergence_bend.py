import time

import xtrack as xt
import xobjects as xo

ctx=xo.ContextCpu()

def mkbend(length=1,angle=0.1,steps=60):
    dw=[ 1/(steps+1) for _ in range(steps+1)]
    kw=[ 1/steps for _ in range(steps)]
    el=[xt.Drift(length=length*dw[0])]
    for dd,kk in zip(dw,kw):
        el.append(xt.Multipole(knl=[angle*kk],hxl=angle*kk,length=length*kk))
        el.append(xt.Drift(length=length*dd))
    return el

t1 = time.perf_counter()
for step in range(1,20):

    elements=mkbend(1,0.1,step)
    line=xt.Line(elements,
                 particle_ref=xt.Particles(mass0=xt.ELECTRON_MASS_EV,p0c=100e9,_context=ctx))
    line.build_tracker(_context=ctx)
    line.configure_radiation(model='mean')
    pp=line.build_particles()
    line.track(pp)
    print(step)
    print(f"px = {pp.px[0]}")
    print(f"py = {pp.py[0]}")
    print(f"pt = {pp.ptau[0]}")
t2 = time.perf_counter()
print(f"Time: {t2-t1}")

elements=mkbend(1,0.1, 2)
line=xt.Line(elements,
                particle_ref=xt.Particles(mass0=xt.ELECTRON_MASS_EV,p0c=100e9,_context=ctx))
line.build_tracker(_context=ctx)
line.configure_radiation(model='mean')

from xtrack.prebuild_kernels import get_suitable_kernel
get_suitable_kernel(
    config=line.config,
    line_element_classes=line.tracker.line_element_classes,
    verbose=True)

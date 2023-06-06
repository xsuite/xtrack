import xtrack as xt
import xpart as xp
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

# for step in range(1,20):

#     elements=mkbend(1,0.1,step)
#     line=xt.Line(elements,
#                  particle_ref=xp.Particles(mass0=xp.ELECTRON_MASS_EV,p0c=100e9,_context=ctx))
#     line.build_tracker(_context=ctx)
#     line.configure_radiation(model='mean')

    # pp=line.build_particles()

    # line.track(pp)
    # line.tracker._current_track_kernel
    # print(step)
    # print(f"px = {pp.px[0]}")
    # print(f"py = {pp.py[0]}")
    # print(f"pt = {pp.ptau[0]}")


elements=mkbend(1,0.1, 2)
line=xt.Line(elements,
                particle_ref=xp.Particles(mass0=xp.ELECTRON_MASS_EV,p0c=100e9,_context=ctx))
line.build_tracker(_context=ctx)
line.configure_radiation(model='mean')

from xtrack.prebuild_kernels import get_suitable_kernel
get_suitable_kernel(
    config=line.config, element_classes=line.tracker.element_classes, verbose=True)

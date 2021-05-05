from pathlib import Path
import xobjects as xo

ctx = xo.ContextPyopencl()

ctx.add_kernels([Path('./source_simple.c')], kernels={})

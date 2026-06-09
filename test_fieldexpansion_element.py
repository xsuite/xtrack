import xtrack as xt
import numpy as np

h = 0.2
a = np.array([[1,0.1], [0.2, 0], [0.3, 0]])
b = np.array([[1,0.1], [0.5, 0]])
bs = np.array([0.1,0])
ny = 5
length=1
fexp = xt.FieldExpansion(length=length, h=h, a=a, b=b, bs=bs, ny=ny)
fexp

line = xt.Line(elements=[fexp])
p = xt.Particles(x=0.01)
line.track(p)
print(p.x)
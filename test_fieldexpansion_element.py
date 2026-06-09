import xtrack as xt
import numpy as np

h = 0.2
a = np.array([[1,0.1,-0.2],[1,0.1,-0.2]])
b = np.array([[1,0.1,-0.2],[1,0.1,-0.2]])
bs = np.array([0.1,0.2,0.1])
ny = 5
length=0.01
fexp = xt.FieldExpansion(length=length, h=h, a=a, b=b, bs=bs, ny=ny)

line = xt.Line(elements=[fexp])
p = xt.Particles(x=0.01)
line.track(p, _force_no_end_turn_actions=True)
print(p.x)
import xtrack as xt
import numpy as np

h = 0.2
a = np.array([[1,1]])
b = np.array([[1,2]])
bs = np.array([1,5])
ny = 5
length=1

line = xt.Line(elements=[xt.FieldExpansion(length=length, h=h, a=a, b=b, bs=bs, ny=ny)])
p = xt.Particles(x=0.01)
line.track(p)
print(p.x)
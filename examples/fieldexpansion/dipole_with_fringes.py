import numpy as np
import xtrack as xt
import matplotlib.pyplot as plt

def get_bshape4(L, bb0, bp0, bbL, bpL, bint):
    """
    Calculate coefficients of a fourth order polynomial with the given values and derivatives
    at the edges, and the given integral
    :param L: length of the segment
    :param bb0: value at 0
    :param bp0: derivative at 0
    :param bbL: value at L
    :param bpL: derivative at L
    :param bint: integral between 0 and L
    :return: set of coefficients, lowest order first
    """
    
    t1 = np.array([1, 0, -18,  32, -15 ])  # p(0)=1
    t2 = np.array([0, 1, -9/2, 6,  -5/2])  # p'(0)=1
    t3 = np.array([0, 0, -12,  28, -15 ])  # p(L)=1
    t4 = np.array([0, 0, 3/2,  -4,  5/2])  # p'(L)=1
    t5 = np.array([0, 0, 30,   -60, 30 ])  # int_0^1 dx p(x)=1

    return (bb0*t1 + L*bp0*t2 + bbL*t3 + L*bpL*t4 + bint/L*t5) * L**np.arange(0, -5, -1)

def calc_value(coeffs, s):
    order = len(coeffs) - 1
    return np.sum(coeffs[:, None] * s[None, :]**np.arange(order + 1)[:, None], axis=0)

gap = 0.05
length = 0.5
bmax = 0.5
h = bmax  # 1 / bending radius

fringe_length = 3*gap
body_length = length - fringe_length

b1_in = get_bshape4(fringe_length, 0, 0, bmax, 0, fringe_length * bmax / 2)
b1_body = np.array([bmax])
b1_out = get_bshape4(fringe_length, bmax, 0, 0, 0, fringe_length * bmax / 2)

fig, ax = plt.subplots()
s1 = np.linspace(0, fringe_length, 100)
ax.plot(s1, calc_value(b1_in, s1))
s2 = np.linspace(0, body_length, 100)
ax.plot(s2+fringe_length, calc_value(b1_body, s2))
s3 = np.linspace(0, fringe_length, 100)
ax.plot(s3+fringe_length+body_length, calc_value(b1_out, s3))

assert body_length >= 0, "Different shape needed to describe such short magnets"

dipole = xt.Line(elements = [
    xt.FieldExpansion(length=fringe_length/2, b=np.array([b1_in]), a=0*np.array([b1_in]), bs=0*b1_in, ny=5, nstep=10),
    xt.FieldExpansion(length=fringe_length/2, b=np.array([b1_in]), a=0*np.array([b1_in]), bs=0*b1_in, ny=5, nstep=10, h=h),
    xt.FieldExpansion(length=body_length, b=np.array([b1_body]), a=0*np.array([b1_body]), bs=0*b1_body, ny=5, nstep=10, h=h),
    xt.FieldExpansion(length=fringe_length/2, b=np.array([b1_out]), a=0*np.array([b1_out]), bs=0*b1_out, ny=5, nstep=10, h=h),
    xt.FieldExpansion(length=fringe_length/2, b=np.array([b1_out]), a=0*np.array([b1_out]), bs=0*b1_out, ny=5, nstep=10)
])

p0 = xt.Particles()
dipole.track(p0)
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
length = 0.5 # magnetic length
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

env=xt.Environment()
nstep=5
env.elements['e0']=xt.FieldExpansion(pkin_const=0, length=fringe_length/2, b=np.array([b1_in]),   a=0*np.array([b1_in]),   bs=0*b1_in,   ny=5, nstep=nstep)
env.elements['e1']=xt.FieldExpansion(pkin_const=0, length=fringe_length/2, b=np.array([b1_in]),   a=0*np.array([b1_in]),   bs=0*b1_in,   ny=5, nstep=nstep, h=h, sstart=fringe_length/2)
env.elements['e2']=xt.FieldExpansion(pkin_const=0, length=body_length,     b=np.array([b1_body]), a=0*np.array([b1_body]), bs=0*b1_body, ny=5, nstep=nstep, h=h)
env.elements['e3']=xt.FieldExpansion(pkin_const=0, length=fringe_length/2, b=np.array([b1_out]),  a=0*np.array([b1_out]),  bs=0*b1_out,  ny=5, nstep=nstep, h=h)
env.elements['e4']=xt.FieldExpansion(pkin_const=0, length=fringe_length/2, b=np.array([b1_out]),  a=0*np.array([b1_out]),  bs=0*b1_out,  ny=5, nstep=nstep, sstart=fringe_length/2)
dipole = env.new_line(name="dipole",components=['e0', 'e1', 'e2', 'e3', 'e4'])

p0 = xt.Particles()
dipole.track(p0)

### Fodo
env['k1quad'] = 1.6
env['lengthquad'] = 0.2
env.new("quad1", xt.Quadrupole, k1='k1quad', length='lengthquad')
env.new("quad2", xt.Quadrupole, k1='-k1quad', length='lengthquad')

fodo= env.new_line(name='fodo', length=5,
                 components=[
                     env.place('quad1', at=0.1),
                     env.place('dipole',  at=1.0),
                     env.place('quad2', at=2.2),
                     env.place('dipole',  at=3.5),
                     ])
fodo.set_particle_ref()
tw = fodo.twiss4d()
tw.plot()
x0,px0=tw.x[0],tw.px[0]
part=fodo.build_particles(x=x0+np.array([0.0,0.01,0.0]),
                          px=px0+np.array([0,0,0]),
                          y=np.array([0.0,0.0,0.01]))
fodo.track(part,num_turns=100000,turn_by_turn_monitor=True)
out=fodo.record_last_track
fig,(a1,a2)=plt.subplots(2,1)

for i in range(len(out.x)):
    a1.plot(fodo.record_last_track.x[i],fodo.record_last_track.px[i],',')
    a2.plot(fodo.record_last_track.y[i],fodo.record_last_track.py[i],',')






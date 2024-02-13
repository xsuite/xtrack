#
# This example creates a simple line with 8 quadrupoles separated by drifts.
# Each quadrupole is split into two thin multipoles, separated by a small drift.
# The first half has one optic and the second half another optic.
#
# Knobs are set for the quadrupoles, and then the second, third, fourth and fifth quads
# are used to match the optics between the two halves.
#

import numpy as np
import xtrack as xt
import xobjects as xo
import matplotlib.pyplot as plt

#### create one FODO line with different optics in first and second half ####
kf1 = 0.05971 / 2
kd1 = -0.05971 / 2
ld1 = 33
lq1 = 1
kf2 = 0.084445 / 2
kd2 = -0.084445 / 2
ld2 = 33
lq2 = 1

line = xt.Line(
    elements=[
        xt.Marker(),
        xt.Drift(length=ld1 / 2),
        xt.Multipole(knl=[0, kf1 / 2], ksl=[0, 0]),
        xt.Drift(length=lq1 / 2),
        xt.Marker(),
        xt.Drift(length=lq1 / 2),
        xt.Multipole(knl=[0, kf1 / 2], ksl=[0, 0]),  # MQD
        xt.Drift(length=ld1),
        xt.Multipole(knl=[0, kd1 / 2], ksl=[0, 0]),
        xt.Drift(length=lq1 / 2),
        xt.Marker(),
        xt.Drift(length=lq1 / 2),
        xt.Multipole(knl=[0, kd1 / 2], ksl=[0, 0]),  # MQD
        xt.Drift(length=ld1),
        xt.Multipole(knl=[0, kf1 / 2], ksl=[0, 0]),
        xt.Drift(length=lq1 / 2),
        xt.Marker(),
        xt.Drift(length=lq1 / 2),
        xt.Multipole(knl=[0, kf1 / 2], ksl=[0, 0]),  # MQD
        xt.Drift(length=ld1),
        xt.Multipole(knl=[0, kd1 / 2], ksl=[0, 0]),
        xt.Drift(length=lq1 / 2),
        xt.Marker(),
        xt.Drift(length=lq1 / 2),
        xt.Multipole(knl=[0, kd1 / 2], ksl=[0, 0]),  # MQF
        xt.Drift(length=ld1 / 2 + ld2 / 2),
        xt.Multipole(knl=[0, kf2 / 2], ksl=[0, 0]),
        xt.Drift(length=lq2 / 2),
        xt.Marker(),
        xt.Drift(length=lq2 / 2),
        xt.Multipole(knl=[0, kf2 / 2], ksl=[0, 0]),  # MQD
        xt.Drift(length=ld2),
        xt.Multipole(knl=[0, kd2 / 2], ksl=[0, 0]),
        xt.Drift(length=lq2 / 2),
        xt.Marker(),
        xt.Drift(length=lq2 / 2),
        xt.Multipole(knl=[0, kd2 / 2], ksl=[0, 0]),  # MQD
        xt.Drift(length=ld2),
        xt.Multipole(knl=[0, kf2 / 2], ksl=[0, 0]),
        xt.Drift(length=lq2 / 2),
        xt.Marker(),
        xt.Drift(length=lq2 / 2),
        xt.Multipole(knl=[0, kf2 / 2], ksl=[0, 0]),  # MQD
        xt.Drift(length=ld2),
        xt.Multipole(knl=[0, kd2 / 2], ksl=[0, 0]),
        xt.Drift(length=lq2 / 2),
        xt.Marker(),
        xt.Drift(length=lq2 / 2),
        xt.Multipole(knl=[0, kd2 / 2], ksl=[0, 0]),  # MQF
        xt.Drift(length=ld2 / 2),
        xt.Marker(),
    ],
    element_names=[
        "start",
        "drift0",
        "mq0..1",
        "drift01",
        "mq0",
        "drift02",
        "mq0..2",
        "drift1",
        "mq1..1",
        "drift11",
        "mq1",
        "drift12",
        "mq1..2",
        "drift2",
        "mq2..1",
        "drift21",
        "mq2",
        "drift22",
        "mq2..2",
        "drift3",
        "mq3..1",
        "drift31",
        "mq3",
        "drift32",
        "mq3..2",
        "drift4",
        "mq4..1",
        "drift41",
        "mq4",
        "drift42",
        "mq4..2",
        "drift5",
        "mq5..1",
        "drift51",
        "mq5",
        "drift52",
        "mq5..2",
        "drift6",
        "mq6..1",
        "drift61",
        "mq6",
        "drift62",
        "mq6..2",
        "drift7",
        "mq7..1",
        "drift71",
        "mq7",
        "drift72",
        "mq7..2",
        "drift8",
        "end",
    ],
)
line.particle_ref = xt.Particles(p0c=450e9, q0=1, mass0=xt.PROTON_MASS_EV)

context = xo.ContextCpu()
line.build_tracker(_context=context)

#### calculate matched twiss parameters ####
Rqf1 = np.array([[1, 0], [kf1 / 2, 1]])
Rqd1 = np.array([[1, 0], [kd1 / 2, 1]])
Rd1 = np.array([[1, ld1], [0, 1]])
Rdmq1 = np.array([[1, lq1 / 2], [0, 1]])
Rcell1f = (
    Rdmq1.dot(Rqf1)
    .dot(Rd1)
    .dot(Rqd1)
    .dot(Rdmq1)
    .dot(Rdmq1)
    .dot(Rqd1)
    .dot(Rd1)
    .dot(Rqf1)
    .dot(Rdmq1)
)
Rcell1d = (
    Rdmq1.dot(Rqd1)
    .dot(Rd1)
    .dot(Rqf1)
    .dot(Rdmq1)
    .dot(Rdmq1)
    .dot(Rqf1)
    .dot(Rd1)
    .dot(Rqd1)
    .dot(Rdmq1)
)
Rqf2 = np.array([[1, 0], [kf2 / 2, 1]])
Rqd2 = np.array([[1, 0], [kd2 / 2, 1]])
Rd2 = np.array([[1, ld2], [0, 1]])
Rdmq2 = np.array([[1, lq2 / 2], [0, 1]])
Rcell2f = (
    Rdmq2.dot(Rqf2)
    .dot(Rd2)
    .dot(Rqd2)
    .dot(Rdmq2)
    .dot(Rdmq2)
    .dot(Rqd2)
    .dot(Rd2)
    .dot(Rqf2)
    .dot(Rdmq2)
)
Rcell2d = (
    Rdmq2.dot(Rqd2)
    .dot(Rd2)
    .dot(Rqf2)
    .dot(Rdmq2)
    .dot(Rdmq2)
    .dot(Rqf2)
    .dot(Rd2)
    .dot(Rqd2)
    .dot(Rdmq2)
)
mu1 = np.arccos(0.5 * (Rcell1f[0, 0] + Rcell1f[1, 1]))
betMax1 = np.sqrt(
    np.abs(Rcell1d[0, 1] * Rcell1d[0, 0] / (Rcell1d[1, 0] * Rcell1d[1, 1]))
)
betMin1 = np.sqrt(
    np.abs(Rcell1f[0, 1] * Rcell1f[1, 1] / (Rcell1f[0, 0] * Rcell1f[1, 0]))
)
mu2 = np.arccos(0.5 * (Rcell2f[0, 0] + Rcell2f[1, 1]))
betMax2 = np.sqrt(
    np.abs(Rcell2d[0, 1] * Rcell2d[0, 0] / (Rcell2d[1, 0] * Rcell2d[1, 1]))
)
betMin2 = np.sqrt(
    np.abs(Rcell2f[0, 1] * Rcell2f[1, 1] / (Rcell2f[0, 0] * Rcell2f[1, 0]))
)

print(
    f"first half: mu: {mu1*180/np.pi:.2f} -- betMax: {betMax1:.2f} -- betMin: {betMin1:.2f}"
)
print(
    f"second half: mu: {mu2*180/np.pi:.2f} -- betMax: {betMax2:.2f} -- betMin: {betMin2:.2f}"
)

#### check twiss ####
tw1s = line.twiss(
    start="mq0",
    end="mq3",
    init=xt.TwissInit(betx=betMax1, bety=betMin1, element_name="mq0", line=line),
)
tw2s = line.twiss(
    start="mq4",
    end="mq7",
    init=xt.TwissInit(betx=betMax2, bety=betMin2, element_name="mq4", line=line),
)
print(tw1s.cols["s", "betx", "bety", "alfx", "alfy"])
print(tw2s.cols["s", "betx", "bety", "alfx", "alfy"])

#### create knobs for the quadrupole strengths ####
line._init_var_management()
kfd_refs = [kf1, kd1, kf1, kd1, kf2, kd2, kf2, kd2]
for i, kfd in enumerate(kfd_refs):
    line.vv["kq" + str(i)] = kfd / 2
    line.element_refs["mq" + str(i) + "..1"].knl[1] = line.vars["kq" + str(i)]
    line.element_refs["mq" + str(i) + "..2"].knl[1] = line.vars["kq" + str(i)]


#### use mq2, mq3, mq4 and mq5 to match the optics between the two halves
opt = line.match(
    solve=False,
    default_tol={None: 5e-10},
    init=tw1s.get_twiss_init("mq0"),
    start=("mq0"),
    end=("end"),
    targets=[
        xt.Target(tar="alfx", value=tw2s.rows["mq6"]["alfx"][0], at="mq6"),
        xt.Target(tar="alfy", value=tw2s.rows["mq6"]["alfy"][0], at="mq6"),
        xt.Target(tar="betx", value=tw2s.rows["mq6"]["betx"][0], at="mq6"),
        xt.Target(tar="bety", value=tw2s.rows["mq6"]["bety"][0], at="mq6"),
    ],
    vary=xt.VaryList(["kq2", "kq3", "kq4", "kq5"]),
)
opt.solve()
print(opt.log())

#### cross-check the final twiss ####
tw3s = line.twiss(
    start="mq0",
    end="end",
    init=xt.TwissInit(betx=betMax1, bety=betMin1, element_name="mq0", line=line),
)
print(tw3s.cols["s", "betx", "bety", "alfx", "alfy"])

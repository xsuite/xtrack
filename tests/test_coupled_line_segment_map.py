import numpy as np
import xobjects as xo
import xtrack as xt
import xpart as xp


# ============================================================
# RANDOM TEST SETTINGS
# ============================================================

N_TESTS = 5
RDT_MAX = 1e-2


# ============================================================
# FCC PARAMETERS (FROM YOU)
# ============================================================

energy = 120
p0c = 120e9
n_ips = 4

beta_x = 240e-3
beta_y = 1.0e-3

alpha_x = 0.0
alpha_y = 0.0

Qx = 398.150 / n_ips
Qy = 398.220 / n_ips
Qs = 0.0334 / n_ips

phi = 15e-3
k2_factor = 0.50

sigma_delta_bs = 0.00176
sigma_z_bs = 5.59e-3
beta_s_bs = sigma_z_bs / sigma_delta_bs


# sextupole optics (can tune later)
beta_x_sext = 3.0
beta_y_sext = 500.0

alpha_x_sext = 0.0
alpha_y_sext = 0.0


context = xo.ContextCpu()


# ============================================================
# RDT → Cbar
# ============================================================

#D. Sagan and D. Rubin. Linear analysis of coupled lattices.
#$Phys. Rev. ST Accel. Beams, 2:074001, Jul 1999.

#R. Calaga, R. Tomás, and A. Franchi. Betatron coupling:
#Merging Hamiltonian and matrix approaches. Phys. Rev. ST
#Accel. Beams, 8:034001, Mar 2005

#Vaibhavi Gawas, Frank Zimmermann, Rogelio Tomas Garcia,
#and Xavier Buffat. MODELLING LINEAR COUPLING
#FOR FCC-ee IN XSUITE . In Proc. IPAC’26.


def rdts_to_Cbar(f1001, f1010):

    re1010, im1010 = np.real(f1010), np.imag(f1010)
    re1001, im1001 = np.real(f1001), np.imag(f1001)

    S11 = (im1010 + im1001)
    S22 = (im1001 - im1010)
    S12 = (-re1010 + re1001)
    S21 = (-re1010 - re1001)

    S = np.array([[S11, S12], [S21, S22]])

    detS = np.linalg.det(S)
    gamma = 1.0 / np.sqrt(1.0 + 4.0 * detS)

    return 2 * gamma * S


def Cbar_to_Cphys(Cbar):

    C11 = Cbar[0, 0] * np.sqrt(beta_x / beta_y)
    C12 = Cbar[0, 1] * np.sqrt(beta_x * beta_y)
    C21 = Cbar[1, 0] / np.sqrt(beta_x * beta_y)
    C22 = Cbar[1, 1] * np.sqrt(beta_y / beta_x)

    return C11, C12, C21, C22


# ============================================================
# TEST LOOP
# ============================================================

for i in range(N_TESTS):

    # -------- random RDTs ----------
    r1001 = np.random.uniform(0, RDT_MAX)
    i1001 = np.random.uniform(0, RDT_MAX)
    r1010 = np.random.uniform(0, RDT_MAX)
    i1010 = np.random.uniform(0, RDT_MAX)

    f1001 = r1001 + 1j * i1001
    f1010 = r1010 + 1j * i1010

    Cbar = rdts_to_Cbar(f1001, f1010)
    c11, c12, c21, c22 = Cbar_to_Cphys(Cbar)

    # -------- sext strengths ----------
    k2_left = k2_factor / (2 * phi * beta_y * beta_y_sext) * np.sqrt(beta_x / beta_x_sext)
    k2_right = k2_left

    # -------- arcs ----------
    arc_left = xt.LineSegmentMap(
        _context=context,
        qx=0.0,
        qy=0.25,
        qs=0.0,
        betx=[beta_x, beta_x_sext],
        bety=[beta_y, beta_y_sext],
        alfx=[alpha_x, alpha_x_sext],
        alfy=[alpha_y, alpha_y_sext],
        c11=[c11, 0.0],
        c12=[c12, 0.0],
        c21=[c21, 0.0],
        c22=[c22, 0.0],
        bets=beta_s_bs,
    )

    arc_mid = xt.LineSegmentMap(
        _context=context,
        qx=Qx,
        qy=Qy - 0.5,
        qs=Qs,
        betx=[beta_x_sext, beta_x_sext],
        bety=[beta_y_sext, beta_y_sext],
        alfx=[alpha_x_sext, alpha_x_sext],
        alfy=[alpha_y_sext, alpha_y_sext],
        bets=beta_s_bs,
    )

    arc_right = xt.LineSegmentMap(
        _context=context,
        qx=0.0,
        qy=0.25,
        qs=0.0,
        betx=[beta_x_sext, beta_x],
        bety=[beta_y_sext, beta_y],
        alfx=[alpha_x_sext, alpha_x],
        alfy=[alpha_y_sext, alpha_y],
        c11=[0.0, c11],
        c12=[0.0, c12],
        c21=[0.0, c21],
        c22=[0.0, c22],
        bets=beta_s_bs,
    )

    sext_left = xt.Multipole(order=2, knl=[0, 0, k2_left], _context=context)
    sext_right = xt.Multipole(order=2, knl=[0, 0, -k2_right], _context=context)

    line = xt.Line(elements=[
        arc_mid,
        sext_right,
        arc_right,
        arc_left,
        sext_left
    ])

    line.particle_ref = xp.Particles(
        _context=context,
        p0c=p0c,
        mass0=xp.ELECTRON_MASS_EV
    )

    line.build_tracker(_context = context, use_prebuilt_kernels = False)

    tw = line.twiss(method="4d", coupling_edw_teng=True)

    f1001_tw = tw.f1001[3]
    f1010_tw = tw.f1010[3]

    print("\n==============================")
    print("TEST", i + 1)

    print("INPUT")
    print("f1001 =", f1001)
    print("f1010 =", f1010)

    print("\nTWISS")
    print("f1001 =", f1001_tw)
    print("f1010 =", f1010_tw)

    ## Xsuite twiss.py outputs -np.conj(rdt) of the rdt which should match the input
    print("\nnegative of Conjugate")
    print("f1001 =",-np.conj(f1001_tw))
    print("f1010 =",-np.conj(f1010_tw))

    
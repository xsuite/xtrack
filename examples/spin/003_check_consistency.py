"""
To intall bmad:
  conda install -c conda-forge bmad
  pip install pytao
"""
from pytao import Tao
import numpy as np
import xtrack as xt

import time

from scipy.constants import c as clight
from scipy.constants import e as qe


def bmad_kicker(By_T, p0c, delta, length, spin_test):

    p_ref = xt.Particles(p0c=p0c, delta=0, mass0=xt.ELECTRON_MASS_EV)
    brho = p_ref.p0c[0] / clight / p_ref.q0

    k0 = By_T / brho

    input = f"""

    bmad_com[spin_tracking_on] = T

    parameter[geometry] = open
    parameter[particle] = positron
    parameter[p0c] = {p0c} ! eV

    particle_start[x] = 0
    particle_start[pz] = {delta} ! this is delta

    particle_start[spin_x] = {spin_test[0]}
    particle_start[spin_y] = {spin_test[1]}
    particle_start[spin_z] = {spin_test[2]}

    beginning[beta_a]  =  1
    beginning[alpha_a]=  0
    beginning[beta_b] =   1
    beginning[alpha_b] =   0
    beginning[eta_x] =  0
    beginning[etap_x] =0

    ! b1: sbend, l={length}, g={k0}! g is h in xtrack
    b1: kicker, l={length}, hkick={-k0 * length}
    dend: drift, l=10.0

    b1[spin_tracking_method] = Symp_Lie_PTC

    myline: line = (b1, dend)

    use, myline
    """

    with open('lattice.bmad', 'w') as f:
        f.write(input)

    time.sleep(1)

    tao = Tao('-lat lattice.bmad -noplot')

    out = tao.orbit_at_s(s_offset=5)

    return out

By_T = 0.023349486663870645
p0c = 700e6
spin_test = [1, 0, 0] # spin vector
length = 0.2
delta = 1e-3

out_on_mom = bmad_kicker(By_T=By_T, p0c=p0c, delta=0, length=length, spin_test=spin_test)
out_off_mom_p0c = bmad_kicker(By_T=By_T, p0c=p0c*(1 + delta), delta=0, length=length,
                              spin_test=spin_test)
out_off_mom_delta = bmad_kicker(By_T=By_T, p0c=p0c, delta=delta, length=length,
                                spin_test=spin_test)
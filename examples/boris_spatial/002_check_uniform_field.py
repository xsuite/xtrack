import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe
from scipy.constants import epsilon_0 as eps0

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField


ctx = xo.ContextCpu()

boris_knl_description = xo.Kernel(
    c_name='boris_step',
    args=[
        xo.Arg(xo.Int64,   name='N_sub_steps'),
        xo.Arg(xo.Float64, name='Dtt'),
        xo.Arg(xo.Float64, name='B_field', pointer=True),
        xo.Arg(xo.Float64, name='B_skew', pointer=True),
        xo.Arg(xo.Float64, name='xn1', pointer=True),
        xo.Arg(xo.Float64, name='yn1', pointer=True),
        xo.Arg(xo.Float64, name='zn1', pointer=True),
        xo.Arg(xo.Float64, name='vxn1', pointer=True),
        xo.Arg(xo.Float64, name='vyn1', pointer=True),
        xo.Arg(xo.Float64, name='vzn1', pointer=True),
        xo.Arg(xo.Float64, name='Ex_n', pointer=True),
        xo.Arg(xo.Float64, name='Ey_n', pointer=True),
        xo.Arg(xo.Float64, name='Bx_n_custom', pointer=True),
        xo.Arg(xo.Float64, name='By_n_custom', pointer=True),
        xo.Arg(xo.Float64, name='Bz_n_custom', pointer=True),
        xo.Arg(xo.Int64,   name='custom_B'),
        xo.Arg(xo.Int64,   name='N_mp'),
        xo.Arg(xo.Int64,   name='N_multipoles'),
        xo.Arg(xo.Float64, name='charge'),
        xo.Arg(xo.Float64, name='mass', pointer=True),
    ],
)

ctx.add_kernels(
    kernels={'boris': boris_knl_description},
    sources=[xt._pkg_root / '_temp/boris_and_solenoid_map/boris.h'],
)

delta=np.array([0, 4])
p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                energy0=1e8,
                px=-1e-3*(1+delta),
                x=[-1e-3, -1e-3],
                y=1e-3,
                delta=delta
                )

p = p0.copy()

class UniformField:
    def __init__(self, B0):
        self.B0 = B0
        self.L = 30
        self.z0 = 15

    def get_field(self, x, y, z):
        Bx = 0 * x
        By = 0 * x
        Bz = 0 * x + self.B0
        return np.array([Bx, By, Bz])

sf = UniformField(B0=1.5)

dt = 1e-10
n_steps = 1500

x_log = []
y_log = []
z_log = []
px_log = []
py_log = []
pp_log = []
beta_x_log = []
beta_y_log = []
beta_z_log = []

for ii in range(n_steps):

    x = p.x.copy()
    y = p.y.copy()
    z = p.s.copy()

    gamma = p.energy / p.mass0
    mass0_kg = p.mass0 * qe / clight**2
    charge0_coulomb = p.q0 * qe

    p0c_J = p.p0c * qe

    Pxc_J = p.px * p0c_J
    Pyc_J = p.py * p0c_J
    Pzc_J = np.sqrt((p0c_J*(1 + p.delta))**2 - Pxc_J**2 - Pyc_J**2)

    vx = Pxc_J / clight / (gamma * mass0_kg) # m/s
    vy = Pyc_J / clight / (gamma * mass0_kg) # m/s
    vz = Pzc_J / clight / (gamma * mass0_kg) # m/s

    Bx, By, Bz = sf.get_field(x + vx * dt / 2,
                                y + vy * dt / 2,
                                z + vz * dt / 2)

    ctx.kernels.boris(
            N_sub_steps=1,
            Dtt=dt,
            B_field=np.array([0.]),
            B_skew=np.array([0.]),
            xn1=x,
            yn1=y,
            zn1=z,
            vxn1=vx,
            vyn1=vy,
            vzn1=vz,
            Ex_n=0 * x,
            Ey_n=0 * x,
            Bx_n_custom=Bx,
            By_n_custom=By,
            Bz_n_custom=Bz,
            custom_B=1,
            N_mp=len(x),
            N_multipoles=0,
            charge=charge0_coulomb,
            mass=mass0_kg * gamma,
    )

    p.x = x
    p.y = y
    p.s = z
    p.px = mass0_kg * gamma * vx * clight / p0c_J
    p.py = mass0_kg * gamma * vy * clight / p0c_J
    pz = mass0_kg * gamma * vz * clight / p0c_J
    pp = np.sqrt(p.px**2 + p.py**2 + pz**2)

    beta_x_after = vx / clight
    beta_y_after = vy / clight
    beta_z_after = vz / clight


    x_log.append(p.x.copy())
    y_log.append(p.y.copy())
    z_log.append(p.s.copy())
    px_log.append(p.px.copy())
    py_log.append(p.py.copy())
    pp_log.append(pp)
    beta_x_log.append(beta_x_after)
    beta_y_log.append(beta_y_after)
    beta_z_log.append(beta_z_after)

x_log = np.array(x_log)
y_log = np.array(y_log)
z_log = np.array(z_log)
px_log = np.array(px_log)
py_log = np.array(py_log)
pp_log = np.array(pp_log)
beta_x_log = np.array(beta_x_log)
beta_y_log = np.array(beta_y_log)
beta_z_log = np.array(beta_z_log)


integrator = BorisSpatialIntegrator(fieldmap_callable=sf.get_field,
                                        s_start=0,
                                        s_end=30,
                                        n_steps=15000)
p_boris = p0.copy()
integrator.track(p_boris)

x_log_boris = np.array(integrator.x_log)
y_log_boris = np.array(integrator.y_log)
z_log_boris = np.array(integrator.z_log)
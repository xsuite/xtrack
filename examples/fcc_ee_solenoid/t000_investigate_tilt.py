import numpy as np
import xtrack as xt
import xobjects as xo
from scipy.constants import c as clight
from scipy.constants import e as qe

from cpymad.madx import Madx

fname = 'fccee_z'; pc_gev = 45.6
# fname = 'fccee_t'; pc_gev = 182.5

mad = Madx()
mad.call('../../test_data/fcc_ee/' + fname + '.seq')
mad.beam(particle='positron', pc=pc_gev)
mad.use('fccee_p_ring')

line = xt.Line.from_madx_sequence(mad.sequence.fccee_p_ring, allow_thick=True,
                                  deferred_expressions=True)
line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV,
                                 gamma0=mad.sequence.fccee_p_ring.beam.gamma)
line.cycle('ip.4', inplace=True)
line.append_element(element=xt.Marker(), name='ip.4.l')

Strategy = xt.Strategy
Teapot = xt.Teapot
slicing_strategies = [
    Strategy(slicing=None),  # Default catch-all as in MAD-X
    Strategy(slicing=Teapot(3), element_type=xt.Bend),
    Strategy(slicing=Teapot(3), element_type=xt.CombinedFunctionMagnet),
    # Strategy(slicing=Teapot(50), element_type=xt.Quadrupole), # Starting point
    Strategy(slicing=Teapot(5), name=r'^qf.*'),
    Strategy(slicing=Teapot(5), name=r'^qd.*'),
    Strategy(slicing=Teapot(5), name=r'^qfg.*'),
    Strategy(slicing=Teapot(5), name=r'^qdg.*'),
    Strategy(slicing=Teapot(5), name=r'^ql.*'),
    Strategy(slicing=Teapot(5), name=r'^qs.*'),
    Strategy(slicing=Teapot(10), name=r'^qb.*'),
    Strategy(slicing=Teapot(10), name=r'^qg.*'),
    Strategy(slicing=Teapot(10), name=r'^qh.*'),
    Strategy(slicing=Teapot(10), name=r'^qi.*'),
    Strategy(slicing=Teapot(10), name=r'^qr.*'),
    Strategy(slicing=Teapot(10), name=r'^qu.*'),
    Strategy(slicing=Teapot(10), name=r'^qy.*'),
    Strategy(slicing=Teapot(50), name=r'^qa.*'),
    Strategy(slicing=Teapot(50), name=r'^qc.*'),
    Strategy(slicing=Teapot(20), name=r'^sy\..*'),
    Strategy(slicing=Teapot(30), name=r'^mwi\..*'),
]
line.discard_tracker()
line.slice_thick_elements(slicing_strategies=slicing_strategies)
line.build_tracker()

theta_tilt = 15e-3 # rad
l_solenoid = 4.4
l_beam = l_solenoid / np.cos(theta_tilt)
ds_sol_start = -l_beam / 2
ds_sol_end = +l_beam / 2
ip_sol = 'ip.1'


tt = line.get_table(attr=True)

s_ip = tt['s', ip_sol]

line.discard_tracker()
line.insert_element(name='sol_start_'+ip_sol, element=xt.Marker(),
                    at_s=s_ip + ds_sol_start)
line.insert_element(name='sol_end_'+ip_sol, element=xt.Marker(),
                    at_s=s_ip + ds_sol_end)

sol_start_tilt = xt.YRotation(angle=-theta_tilt * 180 / np.pi)
sol_end_tilt = xt.YRotation(angle=+theta_tilt * 180 / np.pi)
sol_start_shift = xt.XYShift(dx=l_solenoid/2 * np.tan(theta_tilt))
sol_end_shift = xt.XYShift(dx=l_solenoid/2 * np.tan(theta_tilt))

line.element_dict['sol_start_tilt_'+ip_sol] = sol_start_tilt
line.element_dict['sol_end_tilt_'+ip_sol] = sol_end_tilt
line.element_dict['sol_start_shift_'+ip_sol] = sol_start_shift
line.element_dict['sol_end_shift_'+ip_sol] = sol_end_shift

line.element_dict['sol_entry_'+ip_sol] = xt.Marker()
line.element_dict['sol_exit_'+ip_sol] = xt.Marker()
s_sol_slices = np.linspace(-l_solenoid/2, l_solenoid/2, 11)
l_sol_slices = np.diff(s_sol_slices)
s_sol_slices_entry = s_sol_slices[:-1]

sol_slices = []
for ii in range(len(s_sol_slices_entry)):
    # sol_slices.append(xt.Solenoid(length=l_sol_slices[ii], ks=0)) # Off for now
    sol_slices.append(xt.Drift(length=l_sol_slices[ii])) # Off for now

sol_slice_names = []
sol_slice_names.append('sol_entry_'+ip_sol)
for ii in range(len(s_sol_slices_entry)):
    nn = f'sol_slice_{ii}_{ip_sol}'
    line.element_dict[nn] = sol_slices[ii]
    sol_slice_names.append(nn)
sol_slice_names.append('sol_exit_'+ip_sol)

tt = line.get_table()
names_upstream = list(tt.rows[:'sol_start_'+ip_sol].name)
names_downstream = list(tt.rows['sol_end_'+ip_sol:].name[:-1]) # -1 to exclude '_end_point' added by the table

element_names = (names_upstream
                 + ['sol_start_tilt_'+ip_sol, 'sol_start_shift_'+ip_sol]
                 + sol_slice_names
                 + ['sol_end_shift_'+ip_sol, 'sol_end_tilt_'+ip_sol]
                 + names_downstream)

line.element_names = element_names

line.config.XTRACK_USE_EXACT_DRIFTS = True
line.build_tracker()

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

tw = line.twiss(eneloss_and_damping=True, particle_on_co=line.particle_ref.copy())

print(tw.partition_numbers)
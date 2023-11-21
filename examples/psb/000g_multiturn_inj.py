import numpy as np
import pandas as pd
import xtrack as xt
import xobjects as xo

line = xt.Line.from_json('psb_04_with_chicane_corrected_thin.json')
line.build_tracker()
line.t_turn_s = 0 # Reset time!

line.vars['on_chicane_k0'] = 1
line.vars['on_chicane_k2'] = 1
line.vars['on_chicane_beta_corr'] = 1
line.vars['on_chicane_tune_corr'] = 1

# Install monitor at foil
monitor = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=6000, num_particles=1)
line.discard_tracker()
line.insert_element(index='bi1.tstr1l1', element=monitor, name='monitor_at_foil')
line.build_tracker()

fname = 'inj_distrib.dat.txt'
df = pd.read_table(fname, skiprows=3,
    names="x x' y y' z z' Phase Time Energy Loss".split())

kin_energy_ev = df.Energy.values * 1e6
tot_energy_ev = kin_energy_ev + xt.PROTON_MASS_EV
p0c = line.particle_ref.p0c[0]
tot_energy0_ev = line.particle_ref.energy0[0]

ptau = (tot_energy_ev - tot_energy0_ev) / p0c


part_for_injection = xt.Particles(q0=1, mass0=xt.PROTON_MASS_EV, p0c=line.particle_ref.p0c[0],
                                  ptau=ptau)

part_for_injection.x = df.x.values * 1e-3
part_for_injection.y = df.y.values * 1e-3
part_for_injection.zeta = df.z.values * 1e-3
part_for_injection.px = df["x'"].values  * 1e-3 * (1 + part_for_injection.delta)
part_for_injection.py = df["y'"].values  * 1e-3 * (1 + part_for_injection.delta)
part_for_injection.weight = 10


class ParticlesInjectionRand:

    def __init__(self, particles_to_inject, line, element_name, num_particles_to_inject):
        self.particles_to_inject = particles_to_inject.copy()
        self.line = line
        self.element_name = element_name
        self.num_particles_to_inject = num_particles_to_inject

    def track(self, particles):

        self.particles_to_inject.at_turn += 1

        if not self.num_particles_to_inject:
            return

        assert self.element_name in self.line.element_names
        assert self.line[self.element_name] is self
        at_element = self.line.element_names.index(self.element_name)
        s_element = self.line.tracker._tracker_data_base.element_s_locations[at_element]

        # Get random particles to inject
        idx_inject = np.random.default_rng().choice(len(self.particles_to_inject.x),
                                size=self.num_particles_to_inject, replace=False)

        mask_inject = np.zeros(len(self.particles_to_inject.x), dtype=bool)
        mask_inject[idx_inject] = True

        p_inj = self.particles_to_inject.filter(mask_inject)
        p_inj.s = s_element
        p_inj.at_element = at_element
        p_inj.update_p0c_and_energy_deviations(p0c=self.line.particle_ref.p0c[0])

        particles.add_particles(p_inj)

p_injection = ParticlesInjectionRand(particles_to_inject=part_for_injection,
                                        line=line,
                                        element_name='injection',
                                        num_particles_to_inject=7)

line.discard_tracker()
line.insert_element(index='bi1.tstr1l1', element=p_injection, name='injection')


# Add monitor at end line
monitor = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=10, num_particles=100)
line.insert_element(index='p16ring1$end', element=monitor, name='monitor_at_end')

line.build_tracker()

p = line.build_particles(_capacity=100, x=0)
p.state[0] = -500 # kill the particle added by default

intensity = []
line.enable_time_dependent_vars = True
for iturn in range(8):
    intensity.append(p.weight[p.state>0].sum())
    line.track(p, num_turns=1)

assert np.all(np.sum(monitor.state > 0, axis=0)
              == np.array([ 0,  7, 14, 21, 28, 35, 42, 49, 56,  0]))

assert np.all(monitor.at_element[monitor.state > 0] ==
              line.element_names.index('monitor_at_end'))

assert np.allclose(monitor.s[monitor.state > 0],
                   line.get_s_position('monitor_at_end'), atol=1e-12)
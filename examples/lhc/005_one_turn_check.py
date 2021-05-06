from pathlib import Path
import numpy as np

import xtrack as xt
import xobjects as xo
import sixtracktools
import pysixtrack

from make_short_line import make_short_line

short_test = False # Short line (5 elements)

####################
# Choose a context #
####################

context = xo.ContextCpu()
context = xo.ContextCupy()
#context = xo.ContextPyopencl('0.0')

##################
# Get a sequence #
##################

six = sixtracktools.SixInput(".")
sequence = pysixtrack.Line.from_sixinput(six)
if short_test:
    sequence = make_short_line(sequence)

##################
# Build TrackJob #
##################

tracker = xt.Tracker(context=context,
            sequence=sequence,
            particles_class=xt.Particles,
            local_particle_src=None,
            save_source_as='source.c')

######################
# Get some particles #
######################

sixdump = sixtracktools.SixDump101("res/dump3.dat")

# TODO: The two particles look identical, to be checked
part0_pyst = pysixtrack.Particles(**sixdump[0::2][0].get_minimal_beam())
part1_pyst = pysixtrack.Particles(**sixdump[1::2][0].get_minimal_beam())
# Small kick
part1_pyst.x += 1e-4
part1_pyst.y += 1e-4

pysixtrack_particles = [part0_pyst, part1_pyst]

particles = xt.Particles(pysixtrack_particles=pysixtrack_particles,
                         _context=context)
#########
# Track #
#########
n_turns = 10
tracker.track(particles, num_turns=n_turns)

############################
# Check against pysixtrack #
############################
ip_check = 1
vars_to_check = ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']
pyst_part = pysixtrack_particles[ip_check].copy()
for _ in range(n_turns):
    sequence.track(pyst_part)

for vv in vars_to_check:
    pyst_value = getattr(pyst_part, vv)
    xt_value = getattr(particles, vv)[ip_check]
    passed = np.isclose(xt_value, pyst_value, rtol=1e-9, atol=1e-11)
    if not passed:
        print(f'Not passend on var {vv}!\n'
              f'    pyst:   {pyst_value: .7e}\n'
              f'    xtrack: {xt_value: .7e}\n')
        raise ValueError

##############
# Check  ebe #
##############

pyst_part = pysixtrack_particles[ip_check].copy()
vars_to_check = ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']
problem_found = False
for ii, (eepyst, nn) in enumerate(zip(sequence.elements, sequence.element_names)):
    print(f'\nelement {nn}')
    vars_before = {vv :getattr(pyst_part, vv) for vv in vars_to_check}
    particles.set_one_particle_from_pysixtrack(ip_check, pyst_part)

    tracker.track(particles, ele_start=ii, num_elements=1)

    eepyst.track(pyst_part)
    for vv in vars_to_check:
        pyst_change = getattr(pyst_part, vv) - vars_before[vv]
        xt_change = getattr(particles, vv)[ip_check] -vars_before[vv]
        passed = np.isclose(xt_change, pyst_change, rtol=1e-10, atol=1e-14)
        if not passed:
            problem_found = True
            print(f'Not passend on var {vv}!\n'
                  f'    pyst:   {pyst_change: .7e}\n'
                  f'    xtrack: {xt_change: .7e}\n')
            break

    if not passed:
        break
    else:
        print("Check passed!")


if not problem_found:
    print('All passed on context:')
    print(context)


from pathlib import Path
import numpy as np

import xtrack as xt
import xobjects as xo
import sixtracktools
import pysixtrack

from make_short_line import make_short_line
import time

# # Quick test (for debugging)
# short_test = True# Short line (5 elements)
# n_part = 20
# num_turns = int(100)

short_test = False
n_part = 200#00
num_turns = int(100)

####################
# Choose a context #
####################

context = xo.ContextCpu()
#context = xo.ContextCupy()
#context = xo.ContextPyopencl('0.0')

##################
# Get a sequence #
##################

print('Import sequence')
six = sixtracktools.SixInput(".")
sequence = pysixtrack.Line.from_sixinput(six)
if short_test:
    sequence = make_short_line(sequence)

##################
# Build TrackJob #
##################

print('Build tracker')
tracker = xt.Tracker(context=context,
            sequence=sequence,
            particles_class=xt.Particles,
            local_particle_src=None,
            save_source_as='source.c')

######################
# Get some particles #
######################

print('Import particles')
sixdump = sixtracktools.SixDump101("res/dump3.dat")

# TODO: The two particles look identical, to be checked
part0_pyst = pysixtrack.Particles(**sixdump[0::2][0].get_minimal_beam())
part1_pyst = pysixtrack.Particles(**sixdump[1::2][0].get_minimal_beam())
pysixtrack_particles = [part0_pyst, part1_pyst]

dx_array = np.linspace(-1e-4, 1e-4, n_part)
dy_array = np.linspace(-2e-4, 2e-4, n_part)
pysixtrack_particles = []
for ii in range(n_part):
    pp = part1_pyst.copy()
    pp.x += dx_array[ii]
    pp.y += dy_array[ii]
    pysixtrack_particles.append(pp)

particles = xt.Particles(pysixtrack_particles=pysixtrack_particles,
                         _context=context)
#########
# Track #
#########

print('Track!')
print(f'context: {tracker.context}')
t1 = time.time()
tracker.track(particles, num_turns=num_turns)
context.synchronize()
t2 = time.time()
print(f'Time {(t2-t1)*1000:.2f} ms')
print(f'Time {(t2-t1)*1e6/num_turns/n_part:.2f} us/part/turn')


############################
# Check against pysixtrack #
############################

ip_check = n_part//3*2

print(f'\nTest against pysixtrack over {num_turns} turns on particle {ip_check}:')
vars_to_check = ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']
pyst_part = pysixtrack_particles[ip_check].copy()
for iturn in range(num_turns):
    print(f'turn {iturn}/{num_turns}', end='\r', flush=True)
    sequence.track(pyst_part)

for vv in vars_to_check:
    pyst_value = getattr(pyst_part, vv)
    xt_value = getattr(particles, vv)[ip_check]
    passed = np.isclose(xt_value, pyst_value, rtol=1e-9, atol=1e-11)
    if not passed:
        print(f'Not passed on var {vv}!\n'
              f'    pyst:   {pyst_value: .7e}\n'
              f'    xtrack: {xt_value: .7e}\n')
        raise ValueError
    else:
        print(f'Passed on var {vv}!\n'
              f'    pyst:   {pyst_value: .7e}\n'
              f'    xtrack: {xt_value: .7e}\n')

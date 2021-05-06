from pathlib import Path
import numpy as np

import xtrack as xt
import xobjects as xo
import sixtracktools
import pysixtrack

from make_short_line import make_short_line
import time

short_test = False # Short line (5 elements)
n_part = 20000
num_turns = int(1e4)

####################
# Choose a context #
####################

context = xo.ContextCpu()
context = xo.ContextCupy()
context = xo.ContextPyopencl('2.0')

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

particles = xt.Particles(pysixtrack_particles=n_part*[part1_pyst],
                         _context=context)
#########
# Track #
#########

print('Track!')
t1 = time.time()
tracker.track(particles, num_turns=num_turns)
context.synchronize()
t2 = time.time()
print(f'Time {(t2-t1)*1000:.2f} ms')

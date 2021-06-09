import sixtracktools
import pysixtrack

##################
# Get a sequence #
##################

six = sixtracktools.SixInput(".")
sequence = pysixtrack.Line.from_sixinput(six)


######################
# Get some particles #
######################
sixdump = sixtracktools.SixDump101("res/dump3.dat")
part0_pyst = pysixtrack.Particles(**sixdump[0::2][0].get_minimal_beam())

import pickle
with open('line_and_particle.pkl', 'wb') as fid:
    pickle.dump({
        'line': sequence.to_dict(keepextra=True),
        'particle': part0_pyst.to_dict()},
        fid)

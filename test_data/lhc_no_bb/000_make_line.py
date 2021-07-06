import json
import numpy as np

import sixtracktools
import xline

##################
# Get a sequence #
##################

six = sixtracktools.SixInput(".")
sequence = xline.Line.from_sixinput(six)


######################
# Get some particles #
######################
sixdump = sixtracktools.SixDump101("res/dump3.dat")
part0_pyst = xline.Particles(**sixdump[0::2][0].get_minimal_beam())

# Force active state
part0_pyst.state = 1

#import pickle
#with open('line_and_particle.pkl', 'wb') as fid:
#    pickle.dump({
#        'line': sequence.to_dict(keepextra=True),
#        'particle': part0_pyst.to_dict()},
#        fid)

part_dict = xline.Particles(
        **sixdump[0].get_minimal_beam()).to_dict()
part_dict['state'] = 1

import json
class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif np.issubdtype(type(obj), np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)
with open('line_and_particle.json', 'w') as fid:
    json.dump({
        'line': sequence.to_dict(keepextra=True),
        'particle': part0_pyst.to_dict()},
        fid, cls=Encoder)


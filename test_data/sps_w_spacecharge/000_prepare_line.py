import json
from cpymad.madx import Madx
import pysixtrack

p0c = 25.92e9

class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif np.issubdtype(type(obj), np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


mad = Madx()
mad.call('sps_thin.seq')

line_without_spacecharge = pysixtrack.Line.from_madx_sequence(
                                            mad.sequence['sps'])

part = pysixtrack.Particles(p0c=p0c, x=2e-3, y=3e-3)

with open('line_and_particle.json', 'w') as fid:
    json.dump({
        'line': line_without_spacecharge.to_dict(keepextra=True),
        'particle': part.to_dict()},
        fid, cls=Encoder)

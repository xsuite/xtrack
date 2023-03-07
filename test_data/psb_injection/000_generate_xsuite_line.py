import json

from cpymad.madx import Madx
import xpart as xp
import xtrack as xt
import xobjects as xo

mad = Madx()
mad.call('psb_injection.seq')
mad.use('psb')
mad.twiss()

line = xt.Line.from_madx_sequence(mad.sequence['psb'])
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1.,
                                 gamma0=mad.sequence['psb'].beam.gamma)

with open('line_and_particle.json', 'w') as fid:
    json.dump(line.to_dict(), fid, cls=xo.JEncoder)

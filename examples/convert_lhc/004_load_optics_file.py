import xtrack as xt
from xtrack.mad_parser.parse import MadxParser

dct = MadxParser().parse_file('optics.madx')

env = xt.Environment()
env.vars.default = 0.
for vv in dct['vars']:
    env.vars[vv] = dct['vars'][vv]['expr']
env.vars.default = None
import xtrack as xt
from xtrack.mad_parser.parse import MadxParser

dct = MadxParser().parse_file('optics.madx')

env = xt.Environment()
env.vars.default_to_zero=True
for vv in dct['vars']:
    env.vars[vv] = dct['vars'][vv]['expr']
env.vars.default_to_zero=False


tt_vars = env.vars.get_table()
out = []
out.append('# Quadrupole strengths:')

out_dct = {}
for nn in tt_vars.rows['kq[^s].*'].name:
    vv = tt_vars['value', nn]
    ee = tt_vars['expr', nn]
    if ee == 'None':
        out_dct[nn] = vv
    else:
        out_dct[nn] = ee

from pprint import pformat
out.append(pformat(out_dct))


with open('optics.py', 'w') as fid:
    fid.write('\n'.join(out))
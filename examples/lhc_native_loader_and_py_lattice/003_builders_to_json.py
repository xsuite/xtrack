import xtrack as xt
import xobjects as xo
import xdeps as xd
from cpymad.madx import Madx
import numpy as np

fpath = '../../test_data/lhc_2024/lhc.seq'
mode = 'direct' # 'direct' / 'dict' / 'copy'

with open(fpath, 'r') as fid:
    seq_text = fid.read()

assert ' at=' in seq_text
assert ',at=' not in seq_text
assert 'at =' not in seq_text
seq_text = seq_text.replace(' at=', 'at:=')

env = xt.load(string=seq_text, format='madx', reverse_lines=['lhcb2'])
env.lhcb1.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7000e9)
env.lhcb2.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7000e9)

env.vars.load('../../test_data/lhc_2024/injection_optics.madx')

builder = env.lhcb2.builder

def to_dict(self):
    dct = {'__class__': self.__class__.__name__}
    dct['components'] = []

    formatter = xd.refs.CompactFormatter(scope=None)

    for cc in self.components:
        if not isinstance(cc, xt.environment.Place):
            raise NotImplementedError('Only Place components are implemented for now')
        if not isinstance(cc.name, str):
            raise NotImplementedError('Only str places are implemented for now')

        cc_dct = {}
        cc_dct['name'] = cc.name

        if cc.at is not None:
            if xd.refs.is_ref(cc.at):
                cc_dct['at'] = cc.at._formatted(formatter)
            else:
                cc_dct['at'] = cc.at

        if cc.from_ is not None:
            cc_dct['from_'] = cc.from_

        if cc.anchor is not None:
            cc_dct['anchor'] = cc.anchor

        if cc.from_anchor is not None:
            cc_dct['from_anchor'] = cc.from_anchor

        dct['components'].append(cc_dct)

    if self.refer is not None:
        dct['refer'] = self.refer

    if self.length is not None:
        if xd.refs.is_ref(self.length):
            dct['length'] = self.length._formatted(formatter)
        else:
            dct['l'] = self.length

    if self.s_tol is not None:
        dct['s_tol'] = self.s_tol

    if self.mirror:
        dct['mirror'] = self.mirror

    return dct

def from_dict(cls, dct, env):

    dct = dct.copy()
    dct.pop('__class__', None)

    out = cls(env=env)
    components = dct.pop('components')
    for cc in components:
        out.place(**cc)
    for kk, vv in dct.items():
        setattr(out, kk, vv)

    return out


dct = to_dict(builder)
bb = from_dict(xt.Builder, dct, env)

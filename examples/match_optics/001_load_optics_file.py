import xtrack as xt
import numpy as np

collider = xt.Multiline.from_json('hllhc.json')
collider.build_trackers()

from cpymad.madx import Madx
mad = Madx()
mad.options.echo = False
mad.options.info = False
mad.options.warn = False
mad.call('../../../hllhc15/ramp/opt_endoframp_500_1500.madx')

mad.input(
'''
elm: marker;
seq: sequence, l=1;
e:elm, at=0.5;
endsequence;
beam;
use,sequence=seq;
'''
)

defined_vars = set(mad.globals.keys())

xt.general._print.suppress = True
dummy_line = xt.Line.from_madx_sequence(mad.sequence.seq,
                                        deferred_expressions=True)
xt.general._print.suppress = False

collider._xdeps_vref._owner.update(
    {kk: dummy_line._xdeps_vref._owner[kk] for kk in defined_vars})
collider._xdeps_manager.copy_expr_from(dummy_line._xdeps_manager, "vars")

collider._var_sharing.sync()

for nn in defined_vars:
    if (collider._xdeps_vref[nn]._expr is None
        and len(collider._xdeps_vref[nn]._find_dependant_targets()) > 1 # always contain itself
        ):
        collider._xdeps_vref[nn] = collider._xdeps_vref._owner[nn]

collider.vars['on_x1'] = 30
tw = collider.twiss()
assert np.isclose(tw.lhcb1['px', 'ip1'], 30e-6, atol=1e-8, rtol=0)
assert np.isclose(tw.lhcb2['px', 'ip1'], -30e-6, atol=1e-8, rtol=0)
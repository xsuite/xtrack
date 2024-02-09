from cpymad.madx import Madx
import xtrack as xt

test_data_folder = '../../test_data/'
mad = Madx()

mad.call(test_data_folder + 'pimms/PIMM_orig.seq')
mad.call(test_data_folder + 'pimms/betatron.str')
mad.beam(particle='proton', gamma=1.21315778) # 200 MeV
mad.use('pimms')
seq = mad.sequence.pimms
def_expr = True

line = xt.Line.from_madx_sequence(seq, deferred_expressions=def_expr)
line.particle_ref = xt.Particles(gamma0=seq.beam.gamma,
                                 mass0=seq.beam.mass * 1e9,
                                 q0=seq.beam.charge)

sv = line.survey()

tt = line.get_table()
ttsext = tt.rows[tt.element_type == 'Sextupole']

import xplt
import matplotlib.pyplot as plt
plt.close('all')
xplt.FloorPlot(sv, line, labels=ttsext.name)

plt.show()
from cpymad.madx import Madx
import xtrack as xt

madx = Madx()

madx.input('''
b: rbend, l=1.0, angle=0.5;

ss: sequence, l=5.0, refer=centre;
    b, at=2.5;
endsequence;
beam;
use, sequence=ss;
twiss, betx=1, bety=1;
survey;
''')
sv_madx = xt.Table(madx.table.survey, _copy_cols=True)

line = xt.Line.from_madx_sequence(madx.sequence.ss)
line.remove('ss$start')
line.remove('ss$end')
line.particle_ref = xt.Particles(p0c=1e9)
sv_xs = line.survey()

mad_str = line.to_madx_sequence(sequence_name='ggg')
madx2 = Madx()
madx2.input(mad_str)
madx2.input('''
beam;
use, sequence=ggg;
twiss, betx=1, bety=1;
survey;
''')
sv_madx2 = xt.Table(madx2.table.survey, _copy_cols=True)

env2 = xt.load(string=mad_str, format='madx')
sv_xs2 = env2.ggg.survey()

madng = line.to_madng(keep_files=True)

with open('temp_ng.mad', 'w') as fid:
#     fid.write('''
# do	 -- Begin chunk
# t_turn_s = 0.0
# end	 -- End chunk
# do	 -- Begin chunk
# seq_chunk_0 = bline 'seq_chunk_0' {
# drift 'drift_0' { l = 1.994753437347366},
# rbend 'b' { l = 1.0, angle = 0.5, k0 = 0.4948079185090459, true_rbend=true, e1 = 0.0, e2 = 0.0, fint = 0.0, fintx = 0.0, hgap = 0.0, kill_ent_fringe =\ false, kill_exi_fringe =\ false, k1 = 0.0, knl := {0.0,0.0,0.0,0.0,0.0,0.0}, ksl := {0.0,0.0,0.0,0.0,0.0,0.0}, misalign =\ {dx=0.0, dy=0.0}},
# drift 'drift_1' { l = 1.994753437347366}
# }
# end	 -- End chunk
# do	 -- Begin chunk
# seq = sequence 'seq' { refer='centre', seq_chunk_0 }
# seq_chunk_0 = nil
# end	 -- End chunk
#     ''')

    fid.write('''
do	 -- Begin chunk
t_turn_s = 0.0
end	 -- End chunk
do	 -- Begin chunk
seq_chunk_0 = sequence 'seq_chunk_0' {
rbend 'b' { at = 2.5, l = 1.0, angle = 0.5, k0 = 0.4948079185090459, true_rbend=false, e1 = 0.0, e2 = 0.0, fint = 0.0, fintx = 0.0, hgap = 0.0, kill_ent_fringe =\ false, kill_exi_fringe =\ false, k1 = 0.0, knl := {0.0,0.0,0.0,0.0,0.0,0.0}, ksl := {0.0,0.0,0.0,0.0,0.0,0.0}, misalign =\ {dx=0.0, dy=0.0}},
}
end	 -- End chunk
do	 -- Begin chunk
seq = sequence 'seq' { refer='centre', l=5, seq_chunk_0 }
seq_chunk_0 = nil
end	 -- End chunk
    ''')

import pymadng as pg
mng = pg.MAD()
mng.send(f"""
            mad_func = loadfile('temp_ng.mad', nil, MADX)
            assert(mad_func)
            mad_func()
            """)
sv = mng.survey(sequence='MADX.seq')[0].to_df()
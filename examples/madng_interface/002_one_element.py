with open('temp_ng.mad', 'w') as fid:

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
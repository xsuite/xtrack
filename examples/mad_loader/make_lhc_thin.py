from cpymad.madx import Madx

mad = Madx(stdout=None)
mad.call("lhc_sequence.madx")

mad.beam(sequence="lhcb1", bv= 1, particle="proton", energy= 450)
mad.beam(sequence="lhcb2", bv=-1, particle="proton", energy= 450)

mad.call("lhc_optics.madx")

mad.input("""
select, flag=makethin, clear;
select, flag=makethin, class=mb, slice=4;
select, flag=makethin, class=mq, slice=4;
select, flag=makethin, pattern=mqx,  slice=16;
select, flag=makethin, pattern=mbx,  slice=4;
select, flag=makethin, pattern=mbr,   slice=4;
select, flag=makethin, pattern=mbr,   slice=4;
select, flag=makethin, pattern=mqw,   slice=4;
select, flag=makethin, pattern=mqy\.,    slice=4;
select, flag=makethin, pattern=mqm\.,    slice=4;
select, flag=makethin, pattern=mqmc\.,   slice=4;
select, flag=makethin, pattern=mqml\.,   slice=4;
select, flag=makethin, pattern=mqtlh\.,  slice=2;
select, flag=makethin, pattern=mqtli\.,  slice=2;
select, flag=makethin, pattern=mqt\.  ,  slice=2;

use,sequence=lhcb1; makethin,sequence=lhcb1,makedipedge=true,style=teapot,makeendmarkers=true;
use,sequence=lhcb2; makethin,sequence=lhcb2,makedipedge=true,style=teapot,makeendmarkers=true;

call,file="lhc_aperture.madx";

use,sequence=lhcb1;
use,sequence=lhcb2;
save,file=lhc_thin.madx,beam;
""")

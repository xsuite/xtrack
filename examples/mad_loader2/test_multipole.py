from cpymad.madx import Madx

#from import_madx import sequence_to_line

mad = Madx()
mad.call("examples/mad_loader2/test_multipole.madx")

import xtrack.mad_loader2


def test_iter_elements():
    ml=xtrack.mad_loader2.MadLoader(mad.sequence.seq)
    lst=list(ml.iter_elements())
    mp=lst[2]
    assert mp.knl[1]==1

ml=xtrack.mad_loader2.MadLoader(mad.sequence.seq,enable_expressions=False)
line=ml.make_line()
print(line.to_dict())

ml=xtrack.mad_loader2.MadLoader(mad.sequence.seq,enable_expressions=True)
line=ml.make_line()
print(line.elements)
print(line.to_dict())
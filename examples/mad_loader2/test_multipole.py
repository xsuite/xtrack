from cpymad.madx import Madx

mad = Madx()
mad.call("examples/mad_loader2/test_multipole.madx")

import xtrack.mad_loader2

def test_non_zero_index():
    lst=[1,2,3,0,0,0]
    assert xtrack.mad_loader2.non_zero_len([1,2,3,0,0,0])==3

def test_add_lists():
    a=[1,2,3,1,1,1]
    b=[1,1,1,4,5,6]
    c=xtrack.mad_loader2.add_lists(a,b,8)
    assert c==[2, 3, 4, 5, 6, 7, 0, 0]

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
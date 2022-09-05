import numpy as np

import xtrack.mad_loader2

def test_non_zero_index():
    lst=[1,2,3,0,0,0]
    assert xtrack.mad_loader2.non_zero_len([1,2,3,0,0,0])==3

def test_add_lists():
    a=[1,2,3,1,1,1]
    b=[1,1,1,4,5,6]
    c=xtrack.mad_loader2.add_lists(a,b,8)
    assert c==[2, 3, 4, 5, 6, 7, 0, 0]


def test_add_lists():
    a=[1,2,3,1,1,1]
    b=[1,1,1,4,5,6,7,8]
    c=xtrack.mad_loader2.add_lists(a,b,8)
    assert c==[2, 3, 4, 5, 6, 7, 7, 8]

    a=[1,2,3,1,1,1]
    b=[1,1,1,4,5,6,7,8]
    c=xtrack.mad_loader2.add_lists(a,b,10)
    assert c==[2, 3, 4, 5, 6, 7, 7, 8,0,0]

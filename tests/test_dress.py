import numpy as np
from xtrack import dress
import xobjects as xo

def test_dress():
    class ElementData(xo.Struct):
        n = xo.Int32
        b = xo.Float64
        vv = xo.Float64[:]

    class Element(dress(ElementData)):

        def __init__(self, vv):
            self.xoinitalize(n=len(vv), b=np.sum(vv), vv=vv)

    ele = Element([1,2,3])
    assert ele.n == ele._xobject.n == 3
    assert ele.b == ele._xobject.b == 6
    assert ele.vv[1] == ele._xobject.vv[1] == 2

    ele.vv = [7,8,9]
    assert ele.n == ele._xobject.n == 3
    assert ele.b == ele._xobject.b == 6
    assert ele.vv[1] == ele._xobject.vv[1] == 8

    ele.n = 5.
    assert ele.n == ele._xobject.n == 5

    ele.b = 50
    assert ele.b == ele._xobject.b == 50.

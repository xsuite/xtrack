import xobjects as xo

class _FieldOfDressed:
    def __init__(self, name):
        self.name = name

    def __get__(self, container, ContainerType=None):
        return getattr(container._xobject, self.name)

    def __set__(self, container, value):
        setattr(container._xobject, self.name, value)

def dress(XoStruct):

    DressedXStruct = type(
        'Dressed'+XoStruct.__name__,
        (),
        {'XoStruct': XoStruct})

    for ff in XoStruct._fields:
        fname = ff.name
        setattr(DressedXStruct, fname,
                _FieldOfDressed(fname))

    def xoinitalize(self, **kwargs):
        self._xobject = self.XoStruct(**kwargs)

    def myinit(self, **kwargs):
        self.xoinitalize(**kwargs)

    DressedXStruct.xoinitalize = xoinitalize
    DressedXStruct.__init__ = myinit

    return DressedXStruct



class XMultipole(xo.Struct):
    order = xo.Int64
    bal = xo.Float64


class Multipole(dress(XMultipole)):

    def __init__(self, kn, ks, length):

        order = 3
        bal = kn

        self.xoinitalize(order=order, bal=bal)


mul = Multipole(kn=1, ks=2, length=5)

import xobjects as xo
def dress(XoStruct):

    DressedXStruct = type(
        'Dressed'+XoStruct.__name__,
        (),
        {'XoStruct': XoStruct})

    for ff in XoStruct._fields:
        import pdb; pdb.set_trace()
        fname = f'{ff.name}'

        setattr(DressedXStruct, fname,
                property(lambda self: getattr(self._xobject, fname)))
        def thissetter(self, value):
            setattr(self._xobject, fname, value)
        getattr(DressedXStruct, fname).setter(thissetter)

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

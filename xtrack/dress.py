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

    for ff in ['_buffer', '_offset']:
        setattr(DressedXStruct, ff,
                _FieldOfDressed(ff))

    for ff in XoStruct._fields:
        fname = ff.name
        setattr(DressedXStruct, fname,
                _FieldOfDressed(fname))

    def xoinitialize(self, **kwargs):
        self._xobject = self.XoStruct(**kwargs)

    def myinit(self, **kwargs):
        self.xoinitialize(**kwargs)

    DressedXStruct.xoinitialize = xoinitialize
    DressedXStruct.__init__ = myinit

    return DressedXStruct

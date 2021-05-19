class _FieldOfDressed:
    def __init__(self, name):
        self.name = name
        self.isdressed = False
        self.content = None

    def __get__(self, container, ContainerType=None):
        if self.isdressed:
            return self.content
        else:
            return getattr(container._xobject, self.name)

    def __set__(self, container, value):
        if hasattr(value, '_xobject'): # value is a dressed xobject
            self.isdressed = True
            self.content = value
            setattr(container._xobject, self.name, value._xobject)
            getattr(container, self.name)._xobject = getattr(
                                    container._xobject, self.name)
        else:
            self.isdressed = False
            self.content = None
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

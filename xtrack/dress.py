
class _FieldOfDressed:
    def __init__(self, name, XoStruct):
        self.name = name
        self.isnplikearray = False

        fnames = [ff.name for ff in XoStruct._fields]
        if self.name in fnames:
            ftype = getattr(XoStruct, self.name).ftype
            if  hasattr(ftype, '_itemtype'): # is xo object
                if hasattr(ftype._itemtype, '_dtype'): # valid nplike object
                    self.isnplikearray = True

    def __get__(self, container, ContainerType=None):
        if self.isnplikearray:
            return getattr(container._xobject, self.name).to_nplike()
        elif hasattr(container, '_dressed_'+self.name):
            return getattr(container, '_dressed_'+self.name)
        else:
            return getattr(container._xobject, self.name)

    def __set__(self, container, value):
        if self.isnplikearray:
            getattr(container._xobject, self.name).to_nplike()[:] = value
        elif hasattr(value, '_xobject'): # value is a dressed xobject
            setattr(container, '_dressed_' + self.name, value)
            setattr(container._xobject, self.name, value._xobject)
            getattr(container, self.name)._xobject = getattr(
                                    container._xobject, self.name)
        else:
            self.content = None
            setattr(container._xobject, self.name, value)

def dress(XoStruct, rename={}):

    DressedXStruct = type(
        'Dressed'+XoStruct.__name__,
        (),
        {'XoStruct': XoStruct})

    DressedXStruct._rename = rename

    for ff in ['_buffer', '_offset']:
        setattr(DressedXStruct, ff,
                _FieldOfDressed(ff, XoStruct))

    pynames_list = []
    for ff in XoStruct._fields:
        fname = ff.name
        if fname in rename.keys():
            pyname = rename[fname]
        else:
            pyname = fname
        pynames_list.append(pyname)

        setattr(DressedXStruct, pyname,
                _FieldOfDressed(fname, XoStruct))

        DressedXStruct._fields = pynames_list

    def xoinitialize(self, _xobject=None, **kwargs):

        if _xobject is not None:
            self._xobject = _xobject
        else:

            # Handle dressed inputs
            dressed_kwargs = {}
            for kk, vv in kwargs.items():
                if hasattr(vv, '_xobject'): # vv is dressed
                    dressed_kwargs[kk] = vv
                    kwargs[kk] = vv._xobject

            self._xobject = self.XoStruct(**kwargs)

            # Handle dressed inputs
            for kk, vv in dressed_kwargs.items():
                if kk in self._rename.keys():
                    pyname = self._rename[kk]
                else:
                    pyname = kk
                setattr(self, pyname, vv)


    def myinit(self, _xobject=None, **kwargs):
        self.xoinitialize(_xobject=_xobject, **kwargs)

    def to_dict(self):
        out = {}
        for ff in self._fields:
            vv = getattr(self, ff)
            if hasattr(vv, 'to_dict'):
                out[ff] = vv.to_dict()
            else:
                out[ff] = vv
        return out

    def compile_custom_kernels(self, only_if_needed=False):
        context = self._buffer.context

        if only_if_needed:
            all_found = True
            for kk in self.XoStruct.custom_kernels.keys():
                if kk not in context.kernels.keys():
                    all_found = False
                    break
            if all_found:
                return


        context.add_kernels(
            sources=self.XoStruct.extra_sources,
            kernels=self.XoStruct.custom_kernels,
            extra_classes=[self.XoStruct],
            save_source_as='temp.c')

    DressedXStruct.xoinitialize = xoinitialize
    DressedXStruct.compile_custom_kernels = compile_custom_kernels
    DressedXStruct.to_dict = to_dict
    DressedXStruct.__init__ = myinit

    return DressedXStruct


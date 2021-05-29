
class _FieldOfDressed:
    def __init__(self, name, XoStruct):
        self.name = name
        self.isdressed = False
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
        elif self.isdressed:
            return getattr(container, '_dressed_'+self.name)
        else:
            return getattr(container._xobject, self.name)

    def __set__(self, container, value):
        if self.isnplikearray:
            getattr(container._xobject, self.name).to_nplike()[:] = value
        elif hasattr(value, '_xobject'): # value is a dressed xobject
            self.isdressed = True
            setattr(container, '_dressed_' + self.name, value)
            setattr(container._xobject, self.name, value._xobject)
            getattr(container, self.name)._xobject = getattr(
                                    container._xobject, self.name)
        else:
            self.isdressed = False
            self.content = None
            setattr(container._xobject, self.name, value)

def dress(XoStruct, rename={}):

    DressedXStruct = type(
        'Dressed'+XoStruct.__name__,
        (),
        {'XoStruct': XoStruct})

    for ff in ['_buffer', '_offset']:
        setattr(DressedXStruct, ff,
                _FieldOfDressed(ff, XoStruct))

    for ff in XoStruct._fields:
        fname = ff.name
        if fname in rename.keys():
            pyname = rename[fname]
        else:
            pyname = fname

        setattr(DressedXStruct, pyname,
                _FieldOfDressed(fname, XoStruct))

    def xoinitialize(self, **kwargs):
        self._xobject = self.XoStruct(**kwargs)

    def myinit(self, **kwargs):
        self.xoinitialize(**kwargs)

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

        api_conf = {'prepointer': ' /*gpuglmem*/ '} # TODO: remove
        capi_src, _, capi_cdefs = self.XoStruct._gen_c_api(api_conf)

        context.add_kernels(sources=([capi_src]
                + self.XoStruct.extra_sources),
            kernels=self.XoStruct.custom_kernels,
            extra_cdef='\n'.join([capi_cdefs]),
            save_source_as='temp.c')

    DressedXStruct.xoinitialize = xoinitialize
    DressedXStruct.compile_custom_kernels = compile_custom_kernels
    DressedXStruct.__init__ = myinit

    return DressedXStruct


import xobjects as xo
from ..base_element import BeamElement

class BPMethElement(BeamElement):

    _xofields={'params'             : xo.Float64[:][:],
               'multipole_order'    : xo.Int64,
               's_start'            : xo.Float64,
               's_end'              : xo.Float64,
               'n_steps'            : xo.Int64,}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/bpmethelement.h>',
    ]

    def __init__(self,
                 params=None,
                 multipole_order=1,
                 s_start=None,
                 s_end=None,
                 n_steps=None,
                 **kwargs,
    ):
        if params is not None:
            kwargs['params'] = params
        kwargs['multipole_order'] = multipole_order
        if s_start is not None:
            kwargs['s_start'] = s_start
        if s_end is not None:
            kwargs['s_end'] = s_end
        if n_steps is not None:
            kwargs['n_steps'] = n_steps
        super().__init__(**kwargs)


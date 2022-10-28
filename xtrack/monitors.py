# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xobjects as xo

from .base_element import BeamElement
from .general import _pkg_root


def _monitor_init(
    self,
    _context=None,
    _buffer=None,
    _offset=None,
    _xobject=None,
    start_at_turn=None,
    stop_at_turn=None,
    n_repetitions=None,
    repetition_period=None,
    num_particles=None,
    particle_id_range=None,
    auto_to_numpy=True,
):

    if _xobject is not None:
        self.xoinitialize(_xobject=_xobject)
    else:
        if particle_id_range is not None:
            assert num_particles is None
            part_id_start = particle_id_range[0]
            part_id_end = particle_id_range[1]
        else:
            assert num_particles is not None
            part_id_start = 0
            part_id_end = num_particles

        n_part_ids = part_id_end - part_id_start
        assert n_part_ids >= 0

        n_turns = int(stop_at_turn) - int(start_at_turn)

        if repetition_period is not None:
            assert n_repetitions is not None

        if n_repetitions is not None:
            assert repetition_period is not None

        if repetition_period is None:
            repetition_period = -1
            n_repetitions = 1

        n_records = n_turns * n_part_ids * n_repetitions

        data_init = {nn: n_records for tt, nn in
                        self._ParticlesClass._structure["per_particle_vars"]}

        self.xoinitialize(
            _context=_context,
            _buffer=_buffer,
            _offset=_offset,
            start_at_turn=start_at_turn,
            stop_at_turn=stop_at_turn,
            part_id_start=part_id_start,
            part_id_end=part_id_end,
            n_records=n_records,
            n_repetitions=n_repetitions,
            repetition_period=repetition_period,
            data=data_init,
        )

        self._dressed_data = self._ParticlesClass(_xobject=self._xobject.data,
                                                  _no_reorganize=True)
        self.auto_to_numpy = auto_to_numpy

        with self.data._bypass_linked_vars():
            for tt, nn in self._ParticlesClass._structure["per_particle_vars"]:
                getattr(self.data, nn)[:] = 0

class _FieldOfMonitor:
    def __init__(self, name):
        self.name = name

    def __get__(self, container, ContainerType=None):
        vv = getattr(container.data, self.name)
        if container.auto_to_numpy:
            ctx = container._buffer.context
            vv = ctx.nparray_from_context_array(vv)

        n_cols = container.stop_at_turn - container.start_at_turn

        if container.n_repetitions == 1:
            n_rows = container.n_records // n_cols
            return vv.reshape(n_rows, n_cols)
        else:
            n_rows = container.n_records // n_cols // container.n_repetitions
            #return vv.reshape(container.n_repetitions, n_cols, n_rows)
            return vv.reshape(container.n_repetitions, n_rows, n_cols)


def generate_monitor_class(ParticlesClass):

    _xofields = {
        "start_at_turn": xo.Int64,
        "stop_at_turn": xo.Int64,
        'part_id_start': xo.Int64,
        'part_id_end': xo.Int64,
        'ebe_mode': xo.Int64,
        "n_records": xo.Int64,
        "n_repetitions": xo.Int64,
        "repetition_period": xo.Int64,
        "data": ParticlesClass._XoStruct,
    }

    _extra_c_sources = [
        _pkg_root.joinpath("monitors_src/monitors.h")
    ]

    ParticlesMonitorClass = type(
        "ParticlesMonitor",
        (BeamElement,),
        {"_ParticlesClass": ParticlesClass,
        '_xofields': _xofields,
        '_extra_c_sources': _extra_c_sources,
        },
    )

    ParticlesMonitorClass.__init__ = _monitor_init

    per_particle_vars = ParticlesClass._structure["per_particle_vars"]
    for tt, nn in per_particle_vars:
        setattr(ParticlesMonitorClass, nn, _FieldOfMonitor(name=nn))

    for nn in ['pzeta']:
        setattr(ParticlesMonitorClass, nn, _FieldOfMonitor(name=nn))

    return ParticlesMonitorClass

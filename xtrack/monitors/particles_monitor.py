# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xobjects as xo
import xtrack as xt

from ..base_element import BeamElement
from ..general import _pkg_root

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

    '''
    Beam element logging the coordinates of the particles passing through it.

    Parameters
    ----------
    start_at_turn: int
        Turn at which the monitor starts logging the particles coordinates.
    stop_at_turn: int
        Turn at which the monitor stops logging the particles coordinates.
    n_repetitions: int
        Number of times the monitor repeats the logging of the particles.
    repetition_period: int
        Period in number of turns for the repetition of the logging of the particles.
    num_particles: int
        Number of particles to be logged.
    particle_id_range: tuple of int
        Range of particle ids to be logged.
    auto_to_numpy: bool
        If True, the data is automatically converted to numpy arrays when
        accessed.

    '''

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
                        self._ParticlesClass.per_particle_vars}

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
            for tt, nn in self._ParticlesClass.per_particle_vars:
                getattr(self.data, nn)[:] = 0

def monitor_from_dict(cls, dct, **kwargs):
    xobj = cls._XoStruct(**dct, **kwargs)
    return cls(_xobject=xobj)

def auto_to_numpy(self):
    return self.flag_auto_to_numpy != 0

def set_auto_to_numpy(self, flag):
    if flag:
        self.flag_auto_to_numpy = 1
    else:
        self.flag_auto_to_numpy = 0

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


def _monitor_get_backtrack_element(
                    self, _context=None, _buffer=None, _offset=None):

    return xt.Marker(_context=_context, _buffer=_buffer, _offset=_offset)


class ParticlesMonitor(BeamElement):

    _xofields = {
        "start_at_turn": xo.Int64,
        "stop_at_turn": xo.Int64,
        'part_id_start': xo.Int64,
        'part_id_end': xo.Int64,
        'ebe_mode': xo.Int64,
        "n_records": xo.Int64,
        "n_repetitions": xo.Int64,
        "repetition_period": xo.Int64,
        "flag_auto_to_numpy": xo.Int64,
        "data": xt.Particles,
    }

    _extra_c_sources = [
        _pkg_root.joinpath("monitors/particles_monitor.h")
    ]

    behaves_like_drift = True
    has_backtrack = True
    allow_loss_refinement = True
    _ParticlesClass = xt.Particles


ParticlesMonitor.__init__ = _monitor_init
ParticlesMonitor.get_backtrack_element = _monitor_get_backtrack_element
ParticlesMonitor.from_dict = classmethod(monitor_from_dict)

ParticlesMonitor.auto_to_numpy = property(auto_to_numpy, set_auto_to_numpy)

per_particle_vars = xt.Particles.per_particle_vars
for tt, nn in per_particle_vars:
    setattr(ParticlesMonitor, nn, _FieldOfMonitor(name=nn))

for nn in ['pzeta', 'kin_px', 'kin_py', 'kin_ps', 'kin_xprime', 'kin_yprime']:
    setattr(ParticlesMonitor, nn, _FieldOfMonitor(name=nn))





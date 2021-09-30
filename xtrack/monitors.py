import xobjects as xo

from .base_element import dress_element
from .base_element import _cls_check_requires_sync, _cls_do_global_aperture_check
from .general import _pkg_root


def _monitor_init(
    self,
    _context=None,
    _buffer=None,
    _offset=None,
    start_at_turn=None,
    stop_at_turn=None,
    num_particles=None,
    particle_id_range=None,
    auto_to_numpy=True,
):

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
    n_records = n_turns * n_part_ids

    data_init = {
        nn: n_records for tt, nn in self._ParticlesClass._structure["per_particle_vars"]
    }

    self.xoinitialize(
        _context=_context,
        _buffer=_buffer,
        _offset=_offset,
        start_at_turn=start_at_turn,
        stop_at_turn=stop_at_turn,
        part_id_start=part_id_start,
        part_id_end=part_id_end,
        n_records=n_records,
        data=data_init,
    )

    self._dressed_data = self._ParticlesClass(_xobject=self._xobject.data)
    self.auto_to_numpy = auto_to_numpy

    for tt, nn in self._ParticlesClass._structure["per_particle_vars"]:
        getattr(self.data, nn)[:] = 0


class _FieldOfMonitor:
    def __init__(self, name):
        self.name = name

    def __get__(self, container, ContainerType=None):
        vv = getattr(container.data, self.name)
        n_cols = container.stop_at_turn - container.start_at_turn
        n_rows = container.n_records // n_cols
        if container.auto_to_numpy:
            ctx = container._buffer.context
            vv = ctx.nparray_from_context_array(vv)
        return vv.reshape(n_rows, n_cols)


def generate_monitor_class(ParticlesClass):

    ParticlesMonitorDataClass = type(
        "ParticlesMonitorData",
        (xo.Struct,),
        {
            "start_at_turn": xo.Int64,
            "stop_at_turn": xo.Int64,
            "part_id_start": xo.Int64,
            "part_id_end": xo.Int64,
            "n_records": xo.Int64,
            "data": ParticlesClass.XoStruct,
        },
    )

    ParticlesMonitorDataClass.extra_sources = [
        _pkg_root.joinpath("monitors_src/monitors.h")
    ]

    if not hasattr(ParticlesMonitorDataClass, "requires_sync"):
        ParticlesMonitorDataClass.requires_sync = _cls_check_requires_sync

    if not hasattr(ParticlesMonitorDataClass, "requires_global_aperture_check"):
        ParticlesMonitorDataClass.requires_global_aperture_check = (
            _cls_do_global_aperture_check
        )

    ParticlesMonitorClass = type(
        "ParticlesMonitor",
        (dress_element(ParticlesMonitorDataClass),),
        {"_ParticlesClass": ParticlesClass},
    )

    if not hasattr(ParticlesMonitorClass, "_requires_sync"):
        setattr(ParticlesMonitorClass, "_requires_sync", [])

    if not hasattr(ParticlesMonitorClass, "_global_aperture_check"):
        setattr(ParticlesMonitorClass, "_global_aperture_check", False)

    ParticlesMonitorClass.__init__ = _monitor_init

    per_particle_vars = ParticlesClass._structure["per_particle_vars"]
    for tt, nn in per_particle_vars:
        setattr(ParticlesMonitorClass, nn, _FieldOfMonitor(name=nn))

    return ParticlesMonitorClass

import xobjects as xo

from ..dress_element import dress_element
from ..general import _pkg_root

def generate_monitor_class(ParticlesClass):

    ParticlesMonitorDataClass = type(
            'ParticlesMonitorData',
            (xo.Struct),
            {'start_at_turn': xo.Int64,
             'stop_at_turn': xo.Int64,
             'recorded': ParticlesClass.XoStruct})

    ParticlesMonitorDataClass.extra_sources = [
        _pkg_root.joinpath('monitors_src/monitors.h')]

    ParticlesMonitorClass = type(
            'ParticlesMonitor',
            (dress_element(ParticlesMonitorDataClass),)
            {})

    return ParticlesMonitor

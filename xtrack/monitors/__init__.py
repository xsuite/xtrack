from .particles_monitor import ParticlesMonitor
from .last_turns_monitor import LastTurnsMonitor
from .beam_position_monitor import BeamPositionMonitor
from .beam_size_monitor import BeamSizeMonitor
from .beam_profile_monitor import BeamProfileMonitor
from .multi_element_monitor import MultiElementMonitor
from .beam_stats_monitor import BeamStatsMonitor


monitor_classes = (
    ParticlesMonitor,
    LastTurnsMonitor,
    BeamPositionMonitor,
    BeamSizeMonitor,
    BeamProfileMonitor,
    MultiElementMonitor,
    BeamStatsMonitor,
)

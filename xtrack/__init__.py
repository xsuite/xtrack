from .general import _pkg_root

from .dress import dress
from .base_element import dress_element, BeamElement
from .beam_elements import *
from .line import Line
from .particles import Particles
from .tracker import Tracker

from .monitors import generate_monitor_class
ParticlesMonitor = generate_monitor_class(Particles)

def enable_pyheadtail_interface():
    import xtrack.pyheadtail_interface.pyhtxtparticles as pp
    import xtrack as xt
    xt.Particles = pp.PyHtXtParticles

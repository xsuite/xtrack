from .general import _pkg_root

from .base_element import dress_element, BeamElement
from .beam_elements import *
from .line_frozen import LineFrozen
from .line import Line
from .tracker import Tracker
from .loss_location_refinement import LossLocationRefinement

from .monitors import generate_monitor_class
from . import linear_normal_form

import xpart as _xp
ParticlesMonitor = generate_monitor_class(_xp.Particles)

def Particles(*args, **kwargs):
    raise ValueError(
    "`xtrack.Particles` not available anymore, please use `xpart.Particles`")

def enable_pyheadtail_interface(*args, **kwargs):
    raise ValueError(
    "\n`xtrack.enable_pyheadtail_interface` not available anymore,"
    "\nplease use `xpart.enable_pyheadtail_interface`")

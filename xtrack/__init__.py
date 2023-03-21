# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from .general import _pkg_root

from .base_element import BeamElement
from .beam_elements import *
from .random import *
from .tracker_data import TrackerData
from .line import Line, Node
from .tracker import Tracker, freeze_longitudinal
from .match import Vary, Target, OrbitOnly
from .loss_location_refinement import LossLocationRefinement
from .internal_record import (RecordIdentifier, RecordIndex, new_io_buffer,
                             start_internal_logging, stop_internal_logging)
from .pipeline import (PipelineStatus, PipelineMultiTracker, PipelineBranch,
                        PipelineManager)

from .monitors import *
from . import linear_normal_form
from .multiline import Multiline

from .mad_loader import MadLoader

from .multisetter import MultiSetter

# Flags and test functions
from .line import _is_drift, _behaves_like_drift, _is_aperture, _is_thick, _allow_backtrack
from .line import _lines_equal, _apertures_equal
from .loss_location_refinement import _skip_in_loss_location_refinement

import xpart as _xp
ParticlesMonitor = generate_monitor_class(_xp.Particles)

from ._version import __version__

def Particles(*args, **kwargs):
    raise ValueError(
    "`xtrack.Particles` not available anymore, please use `xpart.Particles`")

def enable_pyheadtail_interface(*args, **kwargs):
    raise ValueError(
    "\n`xtrack.enable_pyheadtail_interface` not available anymore,"
    "\nplease use `xpart.enable_pyheadtail_interface`")

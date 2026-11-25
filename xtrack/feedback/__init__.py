# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""Experimental interfaces for numerical feedback.

Use ``import xtrack as xt`` and the ``xt.feedback`` namespace. Placed Feedback
elements are discovered through ``line.feedback`` and run within Line.track.
"""

from .acquisition import Acquisition
from .hardware import PhysicalActuator
from .interface import Feedback, MonitorInput, Output, Parameter, StateVariable
from .output import KnobOutput
from .pipeline import SharedFeedbackBoundary
from .runtime import FeedbackController, FeedbackSystem
from .timing import CommandQueue, DeadlineSchedule, FirstOrderActuator

__all__ = [
    'Acquisition',
    'CommandQueue',
    'DeadlineSchedule',
    'Feedback',
    'FeedbackController',
    'FeedbackSystem',
    'FirstOrderActuator',
    'KnobOutput',
    'MonitorInput',
    'Output',
    'Parameter',
    'PhysicalActuator',
    'SharedFeedbackBoundary',
    'StateVariable',
]

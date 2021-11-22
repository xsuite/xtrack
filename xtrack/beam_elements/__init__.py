from .elements import *
from .apertures import *
from .beam_interaction import BeamInteraction
from ..base_element import BeamElement

element_classes = tuple( v for v in globals().values() if isinstance(v,type) and issubclass(v,BeamElement))

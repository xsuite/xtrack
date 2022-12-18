# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from . import elements as xstelems
from .base_classes import Element
from .be_beamfields import BeamBeam4D, BeamBeam6D, SCQGaussProfile

class_dict = {}
for nn in dir(xstelems):
    oo = getattr(xstelems, nn)
    if isinstance(oo, type):
        cc = oo
        if issubclass(cc, Element):
            class_dict[cc.__name__] = cc
class_dict['BeamBeam4D'] = BeamBeam4D
class_dict['BeamBeamBiGaussian2D'] = BeamBeam4D
class_dict['BeamBeam3D'] = BeamBeam6D
class_dict['BeamBeamBiGaussian3D'] = BeamBeam6D
class_dict['SpaceChargeBiGaussian'] = SCQGaussProfile



class TestLine:
    @classmethod
    def from_dict(cls, dct):

        self = cls(elements=[], element_names=[])

        if isinstance(dct['elements'], dict):
            el_dct_list = list(dct["elements"][nn] for nn in dct["element_names"])
        else:
            el_dct_list = dct["elements"]

        for el in el_dct_list:
            eltype = class_dict[el["__class__"]]
            eldct=el.copy()
            del eldct['__class__']
            newel = eltype.from_dict(eldct)
            self.elements.append(newel)
        self.element_names = dct["element_names"]
        return self

    def __init__(self, elements, element_names):
        self.elements = elements
        self.element_names = element_names

    def track(self, particles):
        for ee in self.elements:
            ee.track(particles)

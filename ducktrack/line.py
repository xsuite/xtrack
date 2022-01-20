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
        for el in dct["elements"]:
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

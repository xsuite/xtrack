# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2023.                 #
# ######################################### #

from typing import Union, Optional, Iterable

CompoundType = Union['SlicedCompound', 'Compound']


class SlicedCompound:
    def __init__(self, elements):
        self.elements = set(elements)

        self.core = self.elements
        self.aperture = set()
        self.entry_transform = set()
        self.exit_transform = set()
        self.entry = set()
        self.exit = set()

    def __repr__(self):
        return f'{type(self).__name__}({self.elements})'

    def to_dict(self):
        return {
            '__class__': 'SlicedCompound',
            'elements': list(self.elements),
        }

    def remove_element(self, element):
        if element in self.elements:
            self.elements.remove(element)
        else:
            raise ValueError(f'{element} not in {self}')

    def copy(self):
        return SlicedCompound(self.elements.copy())


class Compound:
    """A logical beam element that is composed of other elements.

    This models a thick element with the structure:
    ML. entry other, e.g. a marker
    A.  - aperture entry transform
        - aperture
        - aperture exit transform
    TL. core entry transform
    C.  core (can contain edges, fringes, etc.)
    TR. core exit transform
    MR. exit other, e.g. a marker
    where the core entry and exit transforms are expected to cancel out.

    This logical element is useful for slicing thick elements, as it allows
    the transformations and apertures to be correctly applied to the slices.

    The sliced element is expected to have the following structure:
    ML T(EL) D0 A T(S1) D1 A T(S2) ... DN T(E1) MR, where:
    - S1, S2, ... are the slices of the core C, and
    - D0, D1, ... are the drifts between the slices, and
    - EL, ER are the entry and exit edges, and
    - T(x) := TL x TR.
    """
    def __init__(
            self,
            core,
            aperture=(),
            entry_transform=(),
            exit_transform=(),
            entry=None,
            exit_=None,
    ):
        self.core = set(core)
        self.aperture = set(aperture)
        self.entry_transform = set(entry_transform)
        self.exit_transform = set(exit_transform)

        self.entry = self._make_singleton_set(entry)
        self.exit = self._make_singleton_set(exit_)

    def __repr__(self):
        return (
            f'{type(self).__name__}(core={self.core}, '
            f'aperture={self.aperture}, '
            f'entry_transform={self.entry_transform}, '
            f'exit_transform={self.exit_transform}, '
            f'entry={self.entry}, '
            f'exit={self.exit})'
        )

    @property
    def elements(self):
        return (
            self.entry | self.aperture |
            self.entry_transform | self.core | self.exit_transform |
            self.exit
        )

    def to_dict(self):
        return {
            '__class__': 'Compound',
            'core': list(self.core),
            'aperture': list(self.aperture),
            'entry_transform': list(self.entry_transform),
            'exit_transform': list(self.exit_transform),
            'entry': list(self.entry)[0],
            'exit_': list(self.exit)[0],
        }

    def remove_element(self, element):
        if element in self.core:
            self.core.remove(element)
        elif element in self.aperture:
            self.aperture.remove(element)
        elif element in self.entry_transform:
            self.entry_transform.remove(element)
        elif element in self.exit_transform:
            self.exit_transform.remove(element)
        elif element in self.entry:
            self.entry.remove(element)
        elif element in self.exit:
            self.exit.remove(element)
        else:
            raise ValueError(f'Element {element} not found in compound')

    def copy(self):
        return Compound(
            core=self.core.copy(),
            aperture=self.aperture.copy(),
            entry_transform=self.entry_transform.copy(),
            exit_transform=self.exit_transform.copy(),
            entry=self.entry.copy(),
            exit_=self.exit.copy(),
        )

    @staticmethod
    def _make_singleton_set(var):
        if var is None or len(var) == 0:
            return set()

        if isinstance(var, str):
            return {var}

        try:
            var, = var
            return {var}
        except ValueError:
            raise ValueError(f'Expected singleton set, got {var}')


class CompoundContainer:
    """Container for storing compound elements.

    Maintains a bidirectional mapping between compound names and the
    elements that belong to them.
    """
    def __init__(self, compounds=None):
        if compounds is None:
            self._compounds = {}
            self._compound_name_for_element = {}
        else:
            self._compounds = compounds
            self._compound_name_for_element = {}
            for name, compound in compounds.items():
                for component_name in compound.elements:
                    self._compound_name_for_element[component_name] = name

    def __repr__(self):
        return f'CompoundContainer({self._compounds})'

    def compound_for_name(self, compound_name) -> Optional[CompoundType]:
        return self._compounds.get(compound_name)

    def compound_name_for_element(self, element_name) -> Optional[str]:
        return self._compound_name_for_element.get(element_name)

    def define_compound(self, name, compound):
        self._compounds[name] = compound
        for component_name in compound.elements:
            self._compound_name_for_element[component_name] = name

    def remove_compound(self, name):
        for component_name in self._compounds[name].elements:
            del self._compound_name_for_element[component_name]
        del self._compounds[name]

    @property
    def compound_names(self) -> Iterable[str]:
        return self._compounds.keys()

    @classmethod
    def from_dict(cls, compounds_dict):
        compounds = {}
        for nn, ccdd in compounds_dict.items():
            class_name = ccdd.pop('__class__')
            if class_name == 'Compound':
                compound_class = Compound
            elif class_name == 'SlicedCompound':
                compound_class = SlicedCompound
            else:
                raise ValueError(f'Unknown compound class {class_name}')

            compound = compound_class(**ccdd)
            compounds[nn] = compound

        return cls(compounds=compounds)

    def to_dict(self):
        return {
            name: compound.to_dict()
            for name, compound in self._compounds.items()
        }

    def copy(self):
        copied_compounds = {
            name: compound.copy()
            for name, compound in self._compounds.items()
        }
        return self.__class__(compounds=copied_compounds)

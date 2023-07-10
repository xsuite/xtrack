# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2023.                 #
# ######################################### #

from typing import Union

CompoundType = Union['ThinCompound', 'ThickCompound']


class ThinCompound:
    def __init__(self, elements):
        self.elements = list(elements)

        self.core = self.elements
        self.aperture = []
        self.entry_transform = []
        self.exit_transform = []
        self.entry_other = []
        self.exit_other = []

    def __repr__(self):
        return f'ThinCompound({self.elements})'


class ThickCompound:
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
    ML T(EL) T(D0) A T(S1) T(D1) A T(S2) ... T(DN E1) MR, where:
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
            entry_other=(),
            exit_other=(),
    ):
        self.core = list(core)
        self.aperture = list(aperture)
        self.entry_transform = list(entry_transform)
        self.exit_transform = list(exit_transform)
        self.entry_other = list(entry_other)
        self.exit_other = list(exit_other)

    def __repr__(self):
        return (
            f'ThickCompound(core={self.core}, '
            f'aperture={self.aperture}, '
            f'entry_transform={self.entry_transform}, '
            f'exit_transform={self.exit_transform}, '
            f'entry_other={self.entry_other}, '
            f'exit_other={self.exit_other})'
        )

    @property
    def elements(self):
        return (
            self.entry_other + self.aperture +
            self.entry_transform + self.core + self.exit_transform +
            self.exit_other
        )


class CompoundContainer:
    """Container for storing compound elements.

    Maintains a bidirectional mapping between compound names and the
    elements that belong to them.
    """
    def __init__(self, line, compounds=None):
        self._line = line

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

    def subsequence(self, compound_name):
        return self.compound_for_name(compound_name).elements

    def compound_for_name(self, compound_name):
        return self._compounds.get(compound_name)

    def compound_for_element(self, element_name):
        return self._compound_name_for_element.get(element_name)

    def has_element(self, name):
        return name in self._compound_name_for_element

    def define_compound(self, name, compound):
        self._compounds[name] = compound
        for component_name in compound.elements:
            self._compound_name_for_element[component_name] = name

    def remove_compound(self, name):
        for component_name in self._compounds[name].elements:
            del self._compound_name_for_element[component_name]
        del self._compounds[name]

    @property
    def compound_names(self):
        return self._compounds.keys()

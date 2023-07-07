# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2023.                 #
# ######################################### #

class ThinCompound:
    def __init__(self, elements):
        self.elements = set(elements)


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
        self.core = set(core)
        self.aperture = set(aperture)
        self.entry_transform = set(entry_transform)
        self.exit_transform = set(exit_transform)
        self.entry_other = set(entry_other)
        self.exit_other = set(exit_other)

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
            self.core | self.aperture |
            self.entry_transform | self.exit_transform |
            self.entry_other | self.exit_other
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
        """A subsequence of `line.element_names` corresponding to the compound.
        """
        begin, end = None, None
        compound_elements = self._compounds[compound_name].elements
        for i, name in enumerate(self._line.element_names):
            if name in compound_elements and begin is None:
                begin = i
            elif name in compound_elements:
                end = i + 1

        return self._line.element_names[begin:end]

    def compound_for_element(self, name):
        return self._compounds.get(name, None)

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

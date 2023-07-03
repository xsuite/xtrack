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
    - entry other, e.g. a marker
    - aperture entry transform
    - aperture
    - aperture exit transform
    - core entry transform
    - core
    - core exit transform
    - exit other, e.g. a marker
    where the core entry and exit transforms are expected to cancel out.

    This logical element is useful for slicing thick elements, as it allows
    the transformations and apertures to be correctly applied to the slices.
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
        self.core = core
        self.aperture = set(aperture)
        self.entry_transform = set(entry_transform)
        self.exit_transform = set(exit_transform)
        self.entry_other = set(entry_other)
        self.exit_other = set(exit_other)

    @property
    def elements(self):
        return (
            {self.core} | self.aperture |
            self.entry_transform | self.exit_transform |
            self.entry_other | self.exit_other
        )


class CompoundContainer:
    """Container for storing compound elements.

    Maintains a bidirectional mapping between compound names and the
    elements that belong to them.
    """
    def __init__(self, line):
        self._line = line
        self._compound_for_element = {}
        self._elements_for_compound = {}

    def subsequence(self, compound_name):
        """A subsequence of `line.element_names` corresponding to the compound.
        """
        begin, end = None, None
        compound_elements = self._elements_for_compound[compound_name]
        for i, name in enumerate(self._line.element_names):
            if name in compound_elements and begin is None:
                begin = i
            elif name in compound_elements:
                end = i + 1

        return self._line.element_names[begin:end]

    def compound_for_element(self, name):
        return self._compound_for_element.get(name, None)

    def has_element(self, name):
        return name in self._compound_for_element

    def add_to_compound(self, compound_name, component_names):
        self._elements_for_compound[compound_name] |= set(component_names)
        for name in component_names:
            self._compound_for_element[name] = compound_name

    def remove_elements(self, names):
        for name in names:
            containing_compound = self._compound_for_element[name]
            del self._elements_for_compound[containing_compound][name]

    def remove_compound(self, name):
        self.remove_elements(self._elements_for_compound[name])
        del self._elements_for_compound[name]


# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #


def make_short_line(line):
    new_elements = []
    new_names = []
    found_types = []
    for ee, nn in zip(line.elements, line.element_names):
        if ee.__class__ not in found_types:
            new_elements.append(ee)
            new_names.append(nn)
            found_types.append(ee.__class__)
    line.elements = new_elements
    line.element_names = new_names

    return line

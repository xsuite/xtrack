# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #


def _str_to_index(line, ele, allow_end_point=True):
    if allow_end_point and ele == '_end_point':
        return len(line._element_names_unique)
    if isinstance(ele, str):
        if ele not in line._element_names_unique:
            raise ValueError(f'Element {ele} not found in line')
        return line._element_names_unique.index(ele)
    else:
        return ele

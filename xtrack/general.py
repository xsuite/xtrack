# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from pathlib import Path

_pkg_root = Path(__file__).parent.absolute()

class Table(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self

    def to_pandas(self, index=None,columns=None):
        import pandas as pd

        df = pd.DataFrame(self,columns=columns)
        if index is not None:
            df.set_index(index, inplace=True)
        return df

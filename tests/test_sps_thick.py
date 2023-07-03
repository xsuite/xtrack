import pathlib

import numpy as np

from cpymad.madx import Madx

import xpart as xp
import xtrack as xt
import xobjects
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

@for_all_test_contexts
def test_sps_thick(test_context):
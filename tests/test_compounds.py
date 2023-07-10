# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2023.                 #
# ######################################### #

import numpy as np
import xtrack as xt
from cpymad.madx import Madx

from xtrack.compounds import ThinCompound, ThickCompound
from xtrack.slicing import Strategy, Uniform


def test_slicing_preserve_thick_compound_if_unsliced():
    mad = Madx()
    mad.options.rbarc = False
    mad.input(f"""
    ! Make the sequence a bit longer to accommodate rbends
    ss: sequence, l:=2, refer=entry;
        slice: sbend, at=0, l:=1, angle:=0.1, k0:=0.2, k1=0.1;
        keep: sbend, at=1, l:=1, angle:=0.1, k0:=0.4, k1=0;
    endsequence;
    """)
    mad.beam()
    mad.use(sequence='ss')

    line = xt.Line.from_madx_sequence(
        sequence=mad.sequence.ss,
        deferred_expressions=True,
        allow_thick=True,
    )

    line.slice_thick_elements(slicing_strategies=[
        Strategy(Uniform(2), name='slice'),
        Strategy(None, name='keep'),
    ])

    assert isinstance(line.get_compound_by_name('slice'), ThinCompound)
    assert isinstance(line.get_compound_by_name('keep'), ThickCompound)

import sys
import pathlib
import numpy as np

import xtrack as xt
import xpart as xp
import xobjects as xo
import xfields as xf

from cpymad.madx import Madx

test_data_folder = pathlib.Path(xt.__file__).parent.joinpath("../test_data").absolute()

path = test_data_folder.joinpath("hllhc14_input_mad/")

mad_with_errors = Madx(command_log="mad_final.log")
mad_with_errors.call(str(path.joinpath("final_seq.madx")))
mad_with_errors.use(sequence="lhcb1")
mad_with_errors.twiss()
mad_with_errors.readtable(file=str(path.joinpath("final_errors.tfs")), table="errtab")
mad_with_errors.seterr(table="errtab")
mad_with_errors.set(format=".15g")

mad_no_errors = Madx()
mad_no_errors.call(
    str(test_data_folder.joinpath("hllhc15_noerrors_nobb/sequence.madx"))
)
mad_no_errors.use(sequence="lhcb1")
mad_no_errors.globals["vrf400"] = 16
mad_no_errors.globals["lagrf400.b1"] = 0.5
mad_no_errors.twiss()


mad = mad_with_errors
twmad = mad.twiss(chrom=True)

line = xt.Line.from_madx_sequence(mad.sequence.lhcb1, apply_madx_errors=True)

import xtrack.loader_mad

line2 = xtrack.loader_mad.madx_sequence_to_xtrack_line(
    mad.sequence.lhcb1, classes=xtrack
)

line2._apply_madx_errors(mad.sequence.lhcb1)


for ii,(e1, e2) in enumerate(zip(line.elements, line2.elements)):
    if hasattr(e1, "knl"):
        assert len(e1.knl) == len(e2.knl)
        assert np.allclose(e1.knl, e2.knl)
        assert np.allclose(e1.ksl, e2.ksl)


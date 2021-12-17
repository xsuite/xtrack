import sys
import numpy as np
import xtrack as xt

import pymask as pm

Madx = pm.Madxp
mad = Madx(command_log="mad_final.log")
mad.call("final_seq.madx")
mad.use(sequence="lhcb1")
mad.twiss()
mad.readtable(file="final_errors.tfs", table="errtab")
mad.seterr(table="errtab")
mad.set(format=".15g")
mad.twiss(rmatrix = True)


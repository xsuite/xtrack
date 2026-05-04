import cernlayoutdb as layout

lhc = layout.Machine.from_ldb("LHC", "LS3")
lhc.to_pickle("LHC.pickle")

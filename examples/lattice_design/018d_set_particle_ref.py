import xtrack as xt

pdg_id = xt.particles.pdg.get_pdg_id_from_name('positron')
pref = xt.particles.reference_from_pdg_id(pdg_id, p0c=100e9)
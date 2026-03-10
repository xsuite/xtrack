import xtrack as xt

# TODO:
# - [DONE] Remove the gamma function for low order
# - [DONE] Test spin (radiation is together)
# - [DONE] need a property on rel_ref_is_skew
# - [DONE] need method to see the total multipole strength
# - [DONE] need method to extend knl_rel and ksl_rel when needed
# - MAD-X and MAD-NG interface
# - [DONE] Add new parameters to docstring
# - [DONE] Add new parameters to to __repr__
# - Update env.set_multipole_errors_relative
# - Update docs
# - [DONE] Remove rel_ref_is_skew from bends
# - [Checked] RDT calculation
# - [Checked] Radiation integrals
# - [Checked] Polarization calculation
# - [DONE] Backtracking

m = xt.Multipole(knl=[1e-3], ksl=[2e-3])

p0 = xt.Particles(p0c=1e9)

p = p0.copy()
m.track(p)

m.knl_rel = [0.5]
m.rel_ref_is_skew = True
p = p0.copy()
m.track(p)
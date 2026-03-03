import xtrack as xt

# RENAME:
# - main is skew
# - main strength (read only)
# - main order (for multipole)

# TODO:
# - need a property on rel_ref_is_skew
# - need method to see the total multipole strength
# - need method to extend knl_rel and ksl_rel when needed
# - MAD-NG interface
# - Add new parameters to docstring
# - Update env.set_multipole_errors_relative
# - Update docs
# - Remove rel_ref_is_skew from bends

m = xt.Multipole(knl=[1e-3], ksl=[2e-3])

p0 = xt.Particles(p0c=1e9)

p = p0.copy()
m.track(p)

m.knl_rel = [0.5]
m.rel_ref_is_skew = True
p = p0.copy()
m.track(p)
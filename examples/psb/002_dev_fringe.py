import xtrack as xt
import xpart as xp

line_ng = xt.Line(elements=[xt.Fringe(fint=100000, hgap=0.02, k=0.2)])
line_ng.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=2e9)
line_ng.build_tracker()

line_gianni = line_ng.copy()
line_gianni.config.XTRACK_FRINGE_GIANNI = True
line_gianni.build_tracker()

p_ng = xp.Particles(p0c=2e9, x=0.001, y=0.002, px=0.002, py=0.004)
p_gianni = p_ng.copy()

line_ng.track(p_ng)
line_gianni.track(p_gianni)

print(f'y_ng =     {p_ng.y[0]}')
print(f'y_gianni = {p_gianni.y[0]}')

# R_ng = line_ng.compute_one_turn_matrix_finite_differences(particle_on_co=line_ng.build_particles())
# R_gianni = line_gianni.compute_one_turn_matrix_finite_differences(particle_on_co=line_gianni.build_particles())
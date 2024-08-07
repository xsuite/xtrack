import xtrack as xt

lhc = xt.Multiline.from_madx('job.madx')


# mad = Madx()
# mad.call(fname)

# kwargs = dict()

# lines = {}
# for nn in mad.sequence.keys():
#     lines[nn] = xt.Line.from_madx_sequence(
#         mad.sequence[nn],
#         allow_thick=True,
#         deferred_expressions=True,
#         **kwargs)

#     lines[nn].particle_ref = xt.Particles(
#         mass0=mad.sequence[nn].beam.mass*1e9,
#         q0=mad.sequence[nn].beam.charge,
#         gamma0=mad.sequence[nn].beam.gamma)

#     if mad.sequence[nn].beam.bv == -1:
#         lines[nn].twiss_default['reverse'] = True

# multiline = xt.Multiline(lines=lines)

from cpymad.madx import Madx
#from cpymad import libmadx

mad = Madx(stdout=False)
mad.call("examples/mad_loader2/lhc_thin.madx")
mad.use("lhcb1")


from xtrack import MadLoader

ml=MadLoader(mad.sequence.lhcb1)
line=list(ml.iter_elements())

#ml.make_line()


import cProfile
from pstats import Stats

with cProfile.Profile() as pr:
    #list(ml.iter_elements())
    ml.make_line()

with open('profiling_stats.txt', 'w') as stream:
    stats = Stats(pr, stream=stream)
    stats.strip_dirs()
    stats.sort_stats('time')
    stats.dump_stats('.prof_stats')
    stats.print_stats()


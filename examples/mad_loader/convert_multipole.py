import itertools

from cpymad.madx import Madx

mad = Madx()
mad.call("multipole.madx")

from xtrack import MadLoader

def gen_options(opt1):
    for v in itertools.product(*([[False,True]]*len(opt1))):
        yield dict(zip(opt1,v))

opt1=["enable_expressions",
      "enable_errors",
      "enable_apertures"]

opt2=["skip_markers","merge_drifts","merge_multipoles"]


for opt in gen_options(opt1):
    print(opt)
    ml=MadLoader(mad.sequence.seq,**opt)
    line=ml.make_line()
    print(line.element_names)
    print()

for opt in gen_options(opt2):
    print(opt)
    ml=MadLoader(mad.sequence.seq,**opt)
    line=list(ml.iter_elements())
    print(line)
    print()




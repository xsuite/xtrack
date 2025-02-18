import xtrack as xt

mad_source = '''
a = 3;
b = 3*a;  ! <<<< assiged by value
c := a^5; ! <<<< deferred expression

ll: sequence, l = 1.0;
mm: marker, at = 0.5;
endsequence;

'''

env_native = xt.load_madx_lattice(string=mad_source)

from cpymad.madx import Madx
mad = Madx()
mad.input(mad_source)
mad.beam()
mad.use(sequence='ll')
env_cpymad = xt.Line.from_madx_sequence(mad.sequence.ll,
                                        deferred_expressions=True).env

env_native.get_expr('c')
# (vars['a'] ** 5.0) <<<<<<< CORRECT
env_cpymad.get_expr('c')
# is vars['a^5'] <<<<<<<<<<< WRONG!

env_native.get_expr('b')
# (3.0 * vars['a']) <<<<<<< WRONG!, was assinged by value in madx
env_cpymad.get_expr('b')
# None <<<<<<<<<<< CORRECT
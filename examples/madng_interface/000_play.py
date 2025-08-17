from pymadng import MAD

string = """
! MADX:open_env()                 -- this changes the global scope
mad_func = loadfile("myfodo.mad", nil, MADX) -- this returns a function that can be called later
assert(mad_func)                -- check that the function is there
mad_func()                      -- put the stuff from the file into the global scope
! MADX:close_env()
"""
mad = MAD()
mad.send(string)

# print(mad.MADX.qf.knl.eval())

mad.MADX.myseq
mad.MADX.myseq.kind # is 'sequence'

mad.MADX.myseq.beam = mad.beam()  # set the beam parameters
mad['tws'] = mad.twiss(sequence=mad.MADX.myseq, method=4, save="'atentry'")[0]

mad.tws.colnames().eval()
# ['name',
#  'kind',
#  's',
#  'l',
#  'id',
#  'x',
#  'px',
#  'y',
#  'py',
#  't',
#  'pt',
#  'pc',
#  'ktap',
#  'slc',
#  'turn',
#  'tdir',
#  'eidx',
#  ...]

mad.tws.header
# ['name',
#  'type',
#  'origin',
#  'date',
#  'time',
#  'refcol',
#  'direction',
#  'observe',
#  'implicit',
#  'misalign',
#  'radiate',
#  'particle',
#  'energy',
#  'deltap',
#  'lost',
#  'chrom',
#  'coupling',
#  'trkrdt',
#  'length',
#  'q1',
#  'q2',
#  'q3',
# ]

mad.tws.q1

df = mad['tws'].to_df()
df.attrs
# {'name': 'NAME_OF_MY_SEQUENCE',
#  'type': 'twiss',
#  'origin': 'MAD 1.1.2 OSX 64',
#  'date': '05/06/25',
#  'time': '11:35:58',
#  'refcol': 'name',
#  'direction': np.int32(1),
#  'observe': np.int32(0),
#  'implicit': np.False_,
#  'misalign': np.False_,
#  'radiate': np.False_,
#  'particle': 'positron',
#  'energy': np.int32(1),
#  'deltap': np.int32(0),
#  'lost': np.int32(0),
#  'chrom': np.False_,
#  'coupling': np.False_,
#  'trkrdt': np.False_,
#  'length': np.int32(60),
#  'q1': np.float64(0.75),
#  'q2': np.float64(0.75),
#  'q3': np.int32(0),
#  'dq1': np.float64(-0.9549297832269782),
#  'dq2': np.float64(-0.9549297832269776),
#  'dq3': np.int32(0),
#  'alfap': np.float64(-5.2712342615292805e-17),
#  'etap': np.float64(-2.6111992695381473e-07),
#  'gammatr': np.float64(-137734850.2994461),
#  'synch_1': np.int32(0),
#  'synch_2': np.int32(0),
#  'synch_3': np.int32(0),
#  'synch_4': np.int32(0),
#  'synch_5': np.int32(0),
#  'synch_6': np.int32(0),
#  'synch_8': np.int32(0)}


import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(mad.tws.s, mad.tws.beta11, label='beta11')
plt.plot(mad.tws.s, mad.tws.beta22, label='beta22')
plt.legend()
plt.xlabel('s [m]')
plt.ylabel('beta [m]')

plt.show()
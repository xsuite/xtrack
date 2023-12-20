import xtrack as xt

# Load a line and build a tracker
line = xt.Line.from_json('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.build_tracker()

# Match tunes and chromaticities to assigned values
opt = line.match(
    method='4d',
    vary=[
        xt.VaryList(['kqtf.b1', 'kqtd.b1'], step=1e-8, tag='quad'),
        xt.VaryList(['ksf.b1', 'ksd.b1'], step=1e-4, limits=[-0.1, 0.1], tag='sext'),
    ],
    targets = [
        xt.TargetSet(qx=62.315, qy=60.325, tol=1e-4, tag='tune'),
        xt.TargetSet(dqx=10.0, dqy=12.0, tol=0.01, tag='chrom'),
    ])

# Inspect optimization outcome
opt.target_status()
opt.vary_status()
# prints:
#
# Target status:
# id state tag   tol_met      residue current_val target_val description
#  0 ON    tune     True  1.50695e-06      62.315     62.315 'qx', val=62.315, tol=0.0001, weight=10)
#  1 ON    tune     True   1.4591e-06      60.325     60.325 'qy', val=60.325, tol=0.0001, weight=10)
#  2 ON    chrom    True -0.000211381     9.99979         10 'dqx', val=10, tol=0.01, weight=1)      
#  3 ON    chrom    True  -0.00259168     11.9974         12 'dqy', val=12, tol=0.01, weight=1)      
# Vary status:
# id state tag  name    lower_limit  current_val upper_limit val_at_iter_0   step weight
#  0 ON    quad kqtf.b1        None  4.27291e-05        None             0  1e-08      1
#  1 ON    quad kqtd.b1        None -4.27329e-05        None             0  1e-08      1
#  2 ON    sext ksf.b1         -0.1    0.0117773         0.1             0 0.0001      1
#  3 ON    sext ksd.b1         -0.1   -0.0230563         0.1             0 0.0001      1
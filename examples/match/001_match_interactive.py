import xtrack as xt

# Load a line and build a tracker
line = xt.Line.from_json('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.build_tracker()

# Build optimizer object for tunes and chromaticities without performing optimization
opt = line.match(
    solve=False, # <--
    method='4d',
    vary=[
        xt.VaryList(['kqtf.b1', 'kqtd.b1'], step=1e-8, tag='quad'),
        xt.VaryList(['ksf.b1', 'ksd.b1'], step=1e-4, limits=[-0.1, 0.1], tag='sext'),
    ],
    targets = [
        xt.TargetSet(qx=62.315, qy=60.325, tol=1e-6, tag='tune'),
        xt.TargetSet(dqx=10.0, dqy=12.0, tol=0.01, tag='chrom'),
    ])

opt.target_status()
# prints the following (optimization not performed):
#
# Target status:
# id state tag   tol_met     residue current_val target_val description
#  0 ON    tune    False -0.00499997       62.31     62.315 'qx', val=62.315, tol=1e-06, weight=10)
#  1 ON    tune    False      -0.005       60.32     60.325 'qy', val=60.325, tol=1e-06, weight=10)
#  2 ON    chrom   False    -8.09005     1.90995         10 'dqx', val=10, tol=0.01, weight=1)
#  3 ON    chrom   False     -10.057     1.94297         12 'dqy', val=12, tol=0.01, weight=1)

# Disable optimization of chromaticities and usage of sextupole knobs
opt.disable_targets(tag='chrom')
opt.disable_vary(tag='sext')

opt.show()
# prints:
#
# Vary:
# id tag  state description
#  0 quad ON    name='kqtf.b1', limits=None, step=1e-08, weight=1)
#  1 quad ON    name='kqtd.b1', limits=None, step=1e-08, weight=1)
#  2 sext OFF   name='ksf.b1', limits=(-0.1, 0.1), step=0.0001, weight=1)
#  3 sext OFF   name='ksd.b1', limits=(-0.1, 0.1), step=0.0001, weight=1)
# Targets:
# id tag   state description
#  0 tune  ON    'qx', val=62.315, tol=1e-06, weight=10)
#  1 tune  ON    'qy', val=60.325, tol=1e-06, weight=10)
#  2 chrom OFF   'dqx', val=10, tol=0.01, weight=1)
#  3 chrom OFF   'dqy', val=12, tol=0.01, weight=1)

# Solve (for tunes only)
opt.solve()

opt.target_status()
# prints:
# Target status:
# id state tag   tol_met      residue current_val target_val description
#  0 ON    tune     True -4.51905e-12      62.315     62.315 'qx', val=62.315, tol=1e-06, weight=10)
#  1 ON    tune     True  9.23706e-14      60.325     60.325 'qy', val=60.325, tol=1e-06, weight=10)
#  2 OFF   chrom   False            0     1.89374         10 'dqx', val=10, tol=0.01, weight=1)
#  3 OFF   chrom   False            0     1.91882         12 'dqy', val=12, tol=0.01, weight=1)

# Enable all targets and knobs
opt.enable_all_targets()
opt.enable_all_vary()

# Solve (for tunes and chromaticities)
opt.solve()

opt.target_status()
# prints:
# Target status:
# id state tag   tol_met      residue current_val target_val description
#  0 ON    tune     True -5.62885e-10      62.315     62.315 'qx', val=62.315, tol=1e-06, weight=10)
#  1 ON    tune     True  2.67875e-11      60.325     60.325 'qy', val=60.325, tol=1e-06, weight=10)
#  2 ON    chrom    True -0.000156234     9.99984         10 'dqx', val=10, tol=0.01, weight=1)
#  3 ON    chrom    True -9.81714e-07          12         12 'dqy', val=12, tol=0.01, weight=1)

# Change a target value and the corresponding tolerance
opt.targets[1].value = 60.05
opt.targets[1].tol = 1e-10

opt.target_status()
# prints:
# Target status:
# id state tag   tol_met      residue current_val target_val description
#  0 ON    tune     True -5.62885e-10      62.315     62.315 'qx', val=62.315, tol=1e-06, weight=10)
#  1 ON    tune    False        0.275      60.325      60.05 'qy', val=60.05, tol=1e-10, weight=10)
#  2 ON    chrom    True -0.000156234     9.99984         10 'dqx', val=10, tol=0.01, weight=1)
#  3 ON    chrom    True -9.81714e-07          12         12 'dqy', val=12, tol=0.01, weight=1)

# Perform two optimization steps (without checking for convergence)
opt.step(2)

opt.target_status()
# prints (two steps were not enough to reach convergence):
# Target status:
# id state tag   tol_met      residue current_val target_val description
#  0 ON    tune     True  4.55631e-08      62.315     62.315 'qx', val=62.315, tol=1e-06, weight=10)
#  1 ON    tune    False  2.56767e-09       60.05      60.05 'qy', val=60.05, tol=1e-10, weight=10)
#  2 ON    chrom    True -0.000127644     9.99987         10 'dqx', val=10, tol=0.01, weight=1)
#  3 ON    chrom    True -2.44325e-05          12         12 'dqy', val=12, tol=0.01, weight=1)

# Perform additional two steps
opt.step(2)

opt.target_status()
# prints (convergence was reached):
#
# Target status:
# id state tag   tol_met      residue current_val target_val description
#  0 ON    tune     True -4.00533e-11      62.315     62.315 'qx', val=62.315, tol=1e-06, weight=10)
#  1 ON    tune     True  1.42109e-14       60.05      60.05 'qy', val=60.05, tol=1e-10, weight=10)
#  2 ON    chrom    True  3.19579e-07          10         10 'dqx', val=10, tol=0.01, weight=1)
#  3 ON    chrom    True -1.30694e-09          12         12 'dqy', val=12, tol=0.01, weight=1)

# Tag present configuration
opt.tag(tag='my_tag')

# Reload initial configuration
opt.reload(iteration=0)
opt.target_status()
# prints:
#
# Target status:
# id state tag   tol_met     residue current_val target_val description
#  0 ON    tune    False -0.00499997       62.31     62.315 'qx', val=62.315, tol=1e-06, weight=10
#  1 ON    tune    False        0.27       60.32      60.05 'qy', val=60.05, tol=1e-10, weight=10
#  2 ON    chrom   False    -8.09005     1.90995         10 'dqx', val=10, tol=0.01, weight=1
#  3 ON    chrom   False     -10.057     1.94297         12 'dqy', val=12, tol=0.01, weight=1

# Reload tagged configuration
opt.reload(tag='my_tag')
opt.target_status()
# Target status:
# id state tag   tol_met      residue current_val target_val description
#  0 ON    tune     True -4.00533e-11      62.315     62.315 'qx', val=62.315, tol=1e-06, weight=10
#  1 ON    tune     True  1.42109e-14       60.05      60.05 'qy', val=60.05, tol=1e-10, weight=10
#  2 ON    chrom    True  3.19579e-07          10         10 'dqx', val=10, tol=0.01, weight=1
#  3 ON    chrom    True -1.30694e-09          12         12 'dqy', val=12, tol=0.01, weight=1
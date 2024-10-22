import xtrack as xt

# Load a line and build a tracker
line = xt.Line.from_json('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.build_tracker()

# Match tunes and chromaticities to assigned values
opt = line.match(
    solve=False,
    method='4d', # <- passed to twiss
    vary=[
        xt.VaryList(['kqtf.b1', 'kqtd.b1'], step=1e-8, tag='quad'),
        xt.VaryList(['ksf.b1', 'ksd.b1'], step=1e-4, limits=[-0.1, 0.1], tag='sext'),
    ],
    targets = [
        xt.TargetSet(qx=62.315, qy=60.325, tol=1e-6, tag='tune'),
        xt.TargetSet(dqx=10.0, dqy=12.0, tol=0.01, tag='chrom'),
    ])

opt.solve_homotopy()

# Inspect optimization log
opt.log()

# prints:
# iteration       penalty alpha tag           tol_met target_active hit_limits vary_active        vary_0 ...
#         0       12.8203    -1               nnnn    yyyy          nnnn       yyyy                    0
#         1    0.00207829     0               yyyy    yyyy          nnnn       yyyy          4.27447e-06
#         2    0.00207829    -1 Homotopy it 0 yyyy    yyyy          nnnn       yyyy          4.27447e-06
#         3     0.0033512     0               yyyy    yyyy          nnnn       yyyy          8.54459e-06
#         4     0.0033512    -1 Homotopy it 1 yyyy    yyyy          nnnn       yyyy          8.54459e-06
#         5    0.00676554     0               yyyy    yyyy          nnnn       yyyy          1.28198e-05
#         6    0.00676554    -1 Homotopy it 2 yyyy    yyyy          nnnn       yyyy          1.28198e-05
#         7   0.000607601     0               yyyy    yyyy          nnnn       yyyy          1.70899e-05
#         8   0.000607601    -1 Homotopy it 3 yyyy    yyyy          nnnn       yyyy          1.70899e-05
#         9    0.00254206     0               yyyy    yyyy          nnnn       yyyy          2.13608e-05
#        10    0.00254206    -1 Homotopy it 4 yyyy    yyyy          nnnn       yyyy          2.13608e-05
#        11    0.00127848     0               yyyy    yyyy          nnnn       yyyy          2.56333e-05
#        12    0.00127848    -1 Homotopy it 5 yyyy    yyyy          nnnn       yyyy          2.56333e-05
#        13   0.000285469     0               yyyy    yyyy          nnnn       yyyy          2.99055e-05
#        14   0.000285469    -1 Homotopy it 6 yyyy    yyyy          nnnn       yyyy          2.99055e-05
#        15    0.00366699     0               yyyy    yyyy          nnnn       yyyy          3.41767e-05
#        16    0.00366699    -1 Homotopy it 7 yyyy    yyyy          nnnn       yyyy          3.41767e-05
#        17    0.00158214     0               yyyy    yyyy          nnnn       yyyy          3.84465e-05
#        18    0.00158214    -1 Homotopy it 8 yyyy    yyyy          nnnn       yyyy          3.84465e-05
#        19    0.00353517     0               yyyy    yyyy          nnnn       yyyy           4.2715e-05
#        20    0.00353517    -1 Homotopy it 9 yyyy    yyyy          nnnn       yyyy           4.2715e-05

# Inspect optimization outcome
opt.target_status()
opt.vary_status()
# prints:
#
# Target status:
# id state tag   tol_met      residue current_val target_val description
#  0 ON    tune     True  8.53717e-11      62.315     62.315 'qx', val=62.315, tol=1e-06, weight=10)
#  1 ON    tune     True  1.49214e-13      60.325     60.325 'qy', val=60.325, tol=1e-06, weight=10)
#  2 ON    chrom    True  1.22005e-06          10         10 'dqx', val=10, tol=0.01, weight=1)
#  3 ON    chrom    True -1.87538e-09          12         12 'dqy', val=12, tol=0.01, weight=1)
# Vary status:
# id state tag  name    lower_limit  current_val upper_limit val_at_iter_0   step weight
#  0 ON    quad kqtf.b1        None  4.27163e-05        None             0  1e-08      1
#  1 ON    quad kqtd.b1        None -4.27199e-05        None             0  1e-08      1
#  2 ON    sext ksf.b1         -0.1    0.0118965         0.1             0 0.0001      1
#  3 ON    sext ksd.b1         -0.1   -0.0232137         0.1             0 0.0001      1

# Get knob values after optimization
knobs_after_match = opt.get_knob_values()
# contains: {'kqtf.b1': 4.27163e-05,  'kqtd.b1': -4.27199e-05,
#            'ksf.b1': 0.0118965, 'ksd.b1': -0.0232137}

# Get knob values before optimization
knobs_before_match = opt.get_knob_values(iteration=0)
# contains: {'kqtf.b1': 0, 'kqtd.b1': 0, 'ksf.b1': 0, 'ksd.b1': 0}


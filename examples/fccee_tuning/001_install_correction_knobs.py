"""Install correction knobs, record them in line.metadata, and save the line.

Run after 000_install_errors.py. No correction is performed here.
All settings are below; 002_correct.py consumes the saved line.
"""

from pathlib import Path
import argparse

import xtrack as xt

import corr_orbit
import corr_optics
import corr_coupling
import corr_tune
import corr_chromaticity
import corr_second_order_chrom
from correction_workflow import install_nonlinear_switch

HERE = Path(__file__).resolve().parent
INPUT = HERE / 'fccee_z_lcc_errors.json'
OUTPUT = HERE / 'fccee_z_lcc_correction_knobs.json'
TUNE_KNOBS = ['kqf2', 'kqd1']

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, default=INPUT)
parser.add_argument('--output', type=Path, default=OUTPUT)
args = parser.parse_args()
INPUT, OUTPUT = args.input, args.output

# Load the lattice and error metadata needed by the installers.
loaded = xt.load(INPUT)
line = loaded if isinstance(loaded, xt.Line) else loaded.fccee_p_ring
if 'tuning_errors' not in line.metadata:
    raise ValueError('Run 000_install_errors.py first')
if 'correction_knobs' in line.metadata:
    raise ValueError('Correction knobs already installed')
if 'orbit_optics_correction' in line.metadata:
    raise ValueError('Input already corrected: use the output of 000')

# Add a master switch to ramp existing sextupole-and-higher magnetic terms.
install_nonlinear_switch(line)

# Install independent correction circuits on the lattice with errors.
corr_orbit.install_orbit_knobs(line)
corr_optics.install_optics_knobs(line)
corr_coupling.install_coupling_knobs(line)
corr_tune.install_tune_knobs(line, families=TUNE_KNOBS)
corr_chromaticity.install_chromaticity_knobs(line)
corr_second_order_chrom.install_second_order_chrom_knobs(line)

# Complete the metadata after installing all correction circuits.
circuits = line.metadata['correction_knobs']
circuits['switches'].update(
    optics='on_corr_optics',
    coupling='on_coupling_corr',
    tunes='on_tune_corr',
    errors='on_errors',
)

line.to_json(OUTPUT)
print('Saved', OUTPUT)

# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

################################################################################
# Status
################################################################################

# This file is intentionally not an executable example yet.
#
# Examples 002, 003 and 004 already validate the tails of the total-energy and
# element-kick distributions. Another standalone tail Monte Carlo would repeat
# that work without adding a machine-level observable.
#
# Example 007 is reserved for a future lattice-level lifetime comparison. It
# should only become executable when the machine acceptance and loss model are
# sufficiently well defined to support a defensible lifetime statement.

################################################################################
# Required machine definition
################################################################################

# The completed example needs a specific, documented lattice configuration:
#
# - a committed native-Xsuite FCC-ee lattice;
# - the beam energy and radiation configuration;
# - RF settings and the corresponding longitudinal bucket;
# - physical and momentum apertures relevant to the study;
# - collimation or other loss elements where applicable;
# - a clear definition of which particle states count as lifetime losses.
#
# An arbitrary delta threshold must not be presented as a machine lifetime.

################################################################################
# Required tracking comparison
################################################################################

# The completed example should compare only the public radiation models:
#
# - quantum;
# - quantum-kick.
#
# Both models must use equivalent tapered lattices, initial distributions,
# apertures and tracking durations. The comparison should report:
#
# - surviving and lost particle counts as a function of turn;
# - loss locations and loss-state categories;
# - confidence intervals on loss probabilities or inferred lifetimes;
# - whether the available loss count is statistically informative;
# - explicit PASS, FAIL or LOWSTAT results.

################################################################################
# Statistical requirements
################################################################################

# Lifetime losses are rare-event observables. The implementation therefore
# needs enough independent particles or repeated ensembles to resolve the loss
# probability of interest. It should define in the user-parameter section:
#
# - number of particles and independent repeats;
# - number of turns;
# - target relative uncertainty;
# - minimum number of losses required for PASS or FAIL;
# - confidence level and comparison tolerance.
#
# Zero observed losses must result in a confidence bound or LOWSTAT, not an
# assertion that the lifetime is infinite or that the two models agree.

################################################################################
# Required diagnostics
################################################################################

# The eventual example should remain an interactive demonstration and include:
#
# - survival curves for quantum and quantum-kick;
# - cumulative loss probability with statistical uncertainty;
# - loss-location or loss-category summaries;
# - a model ratio or difference with confidence intervals;
# - concise terminal output explaining every acceptance decision.
#
# Wall-clock performance belongs in examples/tracking_time and should not be an
# acceptance criterion here.

################################################################################
# Completion condition
################################################################################

# Implement this file only when the required machine acceptance is available.
# Until then, keeping this documented placeholder is preferable to adding a
# proxy that duplicates examples 002-004 or overstates what has been validated.

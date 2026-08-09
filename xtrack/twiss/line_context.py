# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xtrack as xt  # To avoid circular imports


def _apply_twiss_line_context(
        twiss_context, line, track_flag_updates, line_config_updates, *,
        freeze_longitudinal, freeze_energy):
    """Apply and automatically restore the normalized temporary line state."""

    if freeze_longitudinal:
        twiss_context.enter_context(xt.freeze_longitudinal(line))
    elif freeze_energy:
        if not line._energy_is_frozen():
            twiss_context.enter_context(xt.line._preserve_config(line))
            line.freeze_energy(force=True)  # force is needed for collective lines

    if track_flag_updates:
        twiss_context.enter_context(xt.line._preserve_track_flags(line))
        for flag_name, value in track_flag_updates.items():
            setattr(line.tracker.track_flags, flag_name, value)

    if line_config_updates:
        twiss_context.enter_context(xt.line._preserve_config(line))
        for config_name, value in line_config_updates.items():
            setattr(line.config, config_name, value)

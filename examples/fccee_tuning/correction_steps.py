"""Promote orbit-only Twiss results to full optics when needed."""


def ensure_full_twiss(tw, *, full_twiss):
    # Call only while the line still has the settings used to compute tw.
    # Use the tracked orbit as a guess; keep the ordinary closed-orbit search.
    if 'betx' in tw._col_names:
        return tw
    return full_twiss(tw)

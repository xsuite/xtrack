import numpy as np

import numpy as np
import pandas as pd

import xtrack as xt
import xpart as xp
import xdeps as xd

import matplotlib.pyplot as plt

line = xt.Line.from_json('psb_03_with_chicane_corrected.json')
line.insert_element(element=xt.Marker(), name='mker_match', at_s=79.874)
line.build_tracker()

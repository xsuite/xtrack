import json
import pathlib

import numpy as np
import xtrack as xt
import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(__file__).parent.joinpath("../test_data").absolute()

with open(test_data_folder / "hllhc14_no_errors_with_coupling_knobs/line_b1.json", "r") as fid:
    dct_b1 = json.load(fid)
input_line = xt.Line.from_dict(dct_b1)

# Load line with knobs on correctors only
from cpymad.madx import Madx

mad = Madx()
mad.call(str(test_data_folder / "hllhc14_no_errors_with_coupling_knobs/lhcb1_seq.madx"))
mad.use(sequence="lhcb1")
input_line_co_ref = xt.Line.from_madx_sequence(
    mad.sequence.lhcb1,
    deferred_expressions=True,
    expressions_for_element_types=("kicker", "hkicker", "vkicker"),
)


# Define dummy metadata
example_metadata = {
    "config_knobs_and_tuning": {
        "knob_settings": {
            "on_x1": 135.0,
            "on_sep1": 0.0,
        },
    },
    "qx": {"lhcb1": 62.31, "lhcb2": 62.31},
    "qy": {"lhcb1": 60.32, "lhcb2": 60.32},
    "delta_cmr": 0.0,
    "knob_names": {
        "lhcb1": {
            "q_knob_1": "dqx.b1_sq",
            "q_knob_2": "dqy.b1_sq",
        },
        "lhcb2": {
            "q_knob_1": "dqx.b2_sq",
            "q_knob_2": "dqy.b2_sq",
        },
    },
}


@for_all_test_contexts
def test_multiline_metadata(test_context, tmp_path):
    # Build the collider with the default constructor
    collider = xt.Multiline(
        lines={"lhcb1": input_line.copy(), "lhcb1_co_ref": input_line_co_ref.copy()}
    )
    collider["lhcb1_co_ref"].particle_ref = collider["lhcb1"].particle_ref.copy()

    # Test the dump and load into/from dictionnary without metadata
    collider = xt.Multiline.from_dict(collider.to_dict())

    # Add metadata
    collider.set_metadata(example_metadata)

    # Test the dump and load into/from json with metadata, to ensure no problem with encoding
    collider.to_json(tmp_path / "test_multiline.json")
    collider_copy = xt.Multiline.from_json(tmp_path / "test_multiline.json")
    assert collider.metadata == collider_copy.metadata == example_metadata

    # Ensuire trackers can still be built
    collider.build_trackers(_context=test_context)

import json
import pathlib

import xtrack as xt
from cpymad.madx import Madx
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(__file__).parent.joinpath('../test_data').absolute()


@for_all_test_contexts
def test_multiline_metadata(test_context):
    with open(test_data_folder / 'hllhc14_no_errors_with_coupling_knobs/line_b1.json', 'r') as fid:
        dct_b1 = json.load(fid)
        input_line = xt.Line.from_dict(dct_b1)

    mad = Madx(stdout=False)
    mad.call(str(test_data_folder / 'hllhc14_no_errors_with_coupling_knobs/lhcb1_seq.madx'))
    mad.use(sequence='lhcb1')

    input_line_co_ref = xt.Line.from_madx_sequence(
        mad.sequence.lhcb1,
        deferred_expressions=True,
        expressions_for_element_types=('kicker', 'hkicker', 'vkicker'),
    )

    # Build the collider with the default constructor
    collider = xt.Environment(
        lines={'lhcb1': input_line.copy(), 'lhcb1_co_ref': input_line_co_ref.copy()}
    )
    collider['lhcb1_co_ref'].particle_ref = collider['lhcb1'].particle_ref.copy()

    # Test the dump and load into/from dictionnary without metadata
    collider = xt.Environment.from_dict(collider.to_dict())

    # Add metadata
    collider.metadata = {
        'config_knobs_and_tuning': {
            'knob_settings': {
                'on_x1': 135.0,
            },
        },
        'qx': {'lhcb1': 62.31, 'lhcb2': 62.31},
        'delta_cmr': 0.0,
    }

    # Test the dump and load into/from dictionnary with metadata
    collider_copy = xt.Environment.from_dict(collider.to_dict())
    
    # Assert that both metadata are still identical
    assert collider.metadata == collider_copy.metadata
    
    # Mutate the copy and check that the original is not changed
    collider_copy.metadata['qx']['lhcb1'] = collider.metadata['qx']['lhcb1'] + 1
    assert collider.metadata != collider_copy.metadata
    
    # Ensuire trackers can still be built
    collider.build_trackers(_context=test_context)

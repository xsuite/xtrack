import numpy as np
import pytest

from xtrack._filling_pattern import _FillingPattern


def test_filling_pattern_dense_and_sparse_are_equivalent():
    dense = _FillingPattern.from_inputs(filling_pattern=[1, 0, 1, 0, 0])
    sparse = _FillingPattern.from_inputs(
        filled_slots=[2, 0], num_slots=5)

    assert dense.num_slots == sparse.num_slots == 5
    np.testing.assert_array_equal(dense.filled_slots, [0, 2])
    np.testing.assert_array_equal(sparse.filled_slots, [0, 2])
    np.testing.assert_array_equal(dense.filling_pattern, [1, 0, 1, 0, 0])
    np.testing.assert_array_equal(sparse.filling_pattern, [1, 0, 1, 0, 0])


def test_filling_pattern_does_not_expose_mutable_state():
    source = np.array([2, 0], dtype=np.int64)
    filling = _FillingPattern.from_inputs(filled_slots=source, num_slots=4)
    source[:] = 3

    slots = filling.filled_slots
    pattern = filling.filling_pattern
    slots[:] = 1
    pattern[:] = 0

    np.testing.assert_array_equal(filling.filled_slots, [0, 2])
    np.testing.assert_array_equal(filling.filling_pattern, [1, 0, 1, 0])
    with pytest.raises(AttributeError, match='immutable'):
        filling._num_slots = 3


def test_filling_pattern_normalizes_compact_bunch_selection():
    filling = _FillingPattern.from_inputs(filled_slots=[0, 2, 4])
    np.testing.assert_array_equal(
        filling.normalize_bunch_selection(), [0, 1, 2])
    np.testing.assert_array_equal(
        filling.normalize_bunch_selection([2, 0]), [2, 0])

    with pytest.raises(ValueError, match='outside'):
        filling.normalize_bunch_selection([3])
    with pytest.raises(ValueError, match='duplicates'):
        filling.normalize_bunch_selection([1, 1])


@pytest.mark.parametrize('kwargs, match', [
    ({'filling_pattern': [1], 'filled_slots': [0]}, 'Only one'),
    ({'filling_pattern': [1], 'filling_scheme': [1]}, 'Only one'),
    ({'filling_pattern': [1, 2]}, 'only zero and one'),
    ({'filling_pattern': [[1]]}, 'one-dimensional'),
    ({'filled_slots': [0, 0]}, 'duplicates'),
    ({'filled_slots': [-1]}, 'non-negative'),
    ({'filled_slots': [1.0]}, 'integers'),
    ({'filled_slots': [2], 'num_slots': 2}, 'outside'),
    ({'filling_pattern': [1, 0], 'num_slots': 3}, 'expected 3'),
])
def test_filling_pattern_rejects_invalid_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _FillingPattern.from_inputs(**kwargs)

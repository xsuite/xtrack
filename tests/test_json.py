import io

import numpy as np

from xtrack import json as xjson


def test_complex_scalar_round_trip():
    data = {"value": 1 + 2j}

    buffer = io.StringIO()
    xjson.dump(data, buffer)
    raw = buffer.getvalue()

    buffer.seek(0)
    loaded_from_file = xjson.load(file=buffer)
    assert loaded_from_file == data
    assert isinstance(loaded_from_file["value"], complex)

    loaded_from_string = xjson.load(string=raw)
    assert loaded_from_string == data
    assert isinstance(loaded_from_string["value"], complex)


def test_complex_array_round_trip_preserves_dtype_and_shape(tmp_path):
    array = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex64)

    path = tmp_path / "complex.json"
    xjson.dump({"array": array}, path)

    loaded = xjson.load(path)
    round_trip = loaded["array"]

    assert isinstance(round_trip, np.ndarray)
    assert round_trip.dtype == array.dtype
    np.testing.assert_allclose(round_trip, array)


def test_complex_array_round_trip_with_gzip(tmp_path):
    array = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)

    path = tmp_path / "complex.json.gz"
    xjson.dump({"array": array}, path)

    loaded = xjson.load(path)
    round_trip = loaded["array"]

    assert isinstance(round_trip, np.ndarray)
    assert round_trip.dtype == array.dtype
    np.testing.assert_allclose(round_trip, array)


def test_python_primitive_types_round_trip():
    data = {
        "int": 42,
        "float": 3.14,
        "bool": True,
        "str": "xtrack",
        "list": [1, 2, 3],
        "dict": {"nested": "value"},
        "none": None,
    }

    buffer = io.StringIO()
    xjson.dump(data, buffer)

    buffer.seek(0)
    loaded = xjson.load(file=buffer)
    assert loaded == data


def test_numpy_scalar_round_trip():
    data = {"value": np.int64(5)}

    buffer = io.StringIO()
    xjson.dump(data, buffer)

    buffer.seek(0)
    loaded = xjson.load(file=buffer)

    assert loaded["value"] == 5
    assert isinstance(loaded["value"], int)


def test_numpy_real_array_round_trip():
    array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    buffer = io.StringIO()
    xjson.dump({"array": array}, buffer)

    buffer.seek(0)
    loaded = xjson.load(file=buffer)

    assert loaded["array"] == array.tolist()

import math
import pytest
from app.utils import predict

def test_predict_logic():
    entry_data = [[1, 2, 3], [5], [1.5, 2.5]]
    expected_data = [[2, 4, 6], [10], [3,5]]
    for i in range(len(entry_data)):
        result = predict(entry_data[i])
        assert result == expected_data[i]

def test_predict_limit():
    entry_data = [[0], [], [-2], [-1000], [2000000]]
    expected_data = [[0], [], [-4], [-2000], [4000000]]
    for i in range(len(entry_data)):
        try:
            result = predict(entry_data[i])
            assert result == expected_data[i]
        except Exception as e:
            print(f"Une erreur est survenue:{e}")

def test_invalid_cases():
    invalid_features = [None, "abc", {}, [1, "abc", 3], [1, None, 3]]

    for i in range(len(invalid_features)):
        try:
            result = predict(invalid_features[i])
        except Exception as e:
            print(f"Une erreur est survenue:{e}")

@pytest.mark.parametrize(
        "features, expected",
        [
            ([1, 2, 3], [2, 4, 6]),
            ([5], [10]),
            ([1.5, 2.5], [3,5])
        ],
)

def test_predict_nominal_cases(features, expected):
    result = predict(features)
    assert result == pytest.approx(expected)

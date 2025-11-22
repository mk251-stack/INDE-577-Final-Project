import pytest
import numpy as np
from rice_ml import euclidean_distance, manhattan_distance


def test_euclidean_distance_basic():
    assert euclidean_distance(np.array([0, 0]), np.array([3, 4])) == 5.0
    assert euclidean_distance([1, 2, 3], [1, 2, 3]) == 0.0


def test_manhattan_distance_basic():
    assert manhattan_distance(np.array([1, 2, 3]), np.array([4, 0, 3])) == 5.0
    assert manhattan_distance([0, 0], [0, 0]) == 0.0


def test_invalid_shape():
    with pytest.raises(ValueError):
        euclidean_distance(np.array([[1, 2], [3, 4]]), np.array([1, 2]))
    with pytest.raises(ValueError):
        manhattan_distance(np.array([1, 2, 3]), np.array([1, 2]))


def test_type_validation():
    with pytest.raises(TypeError):
        euclidean_distance(["a", "b"], [1, 2])
    with pytest.raises(TypeError):
        manhattan_distance([1, 2], ["x", "y"])


def test_symmetry_and_nonnegative():
    a, b = np.array([1, 2, 3]), np.array([4, 5, 6])
    assert euclidean_distance(a, b) == euclidean_distance(b, a)
    assert manhattan_distance(a, b) == manhattan_distance(b, a)
    assert euclidean_distance(a, b) >= 0
    assert manhattan_distance(a, b) >= 0

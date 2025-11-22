"""
Distance metrics module.

This module provides common distance functions for numerical vectors,
implemented using NumPy for efficient computation and robust error handling.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np

__all__ = ["euclidean_distance", "manhattan_distance"]


def _to_1d_float_array(x, name: str) -> np.ndarray:
    """
    Convert input to a 1D float NumPy array with clear, consistent errors.

    Parameters
    ----------
    x : array_like
        Input vector.
    name : str
        Name used in error messages ("a" or "b").

    Returns
    -------
    np.ndarray
        1D array of dtype float.

    Raises
    ------
    ValueError
        If the array is not 1-dimensional.
    TypeError
        If the input contains non-numeric elements.
    """
    arr = np.asarray(x)

    # Dimensionality check first (so shape errors surface consistently)
    if arr.ndim != 1:
        raise ValueError(f"Input array '{name}' must be 1-dimensional; got {arr.ndim}D.")

    # If dtype is not numeric (e.g., object from mixed/str), raise TypeError
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"All elements of '{name}' must be numeric (int or float).")

    # Safe cast to float; if this fails, treat as non-numeric
    try:
        arr = arr.astype(float, copy=False)
    except (TypeError, ValueError):
        raise TypeError(f"All elements of '{name}' must be numeric (int or float).")
    return arr


def _validate_arrays(a, b) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and align two 1D arrays for distance computation.
    """
    a_arr = _to_1d_float_array(a, "a")
    b_arr = _to_1d_float_array(b, "b")

    if a_arr.shape != b_arr.shape:
        raise ValueError(f"Arrays must have the same shape: a.shape={a_arr.shape}, b.shape={b_arr.shape}.")

    return a_arr, b_arr


def euclidean_distance(a, b) -> float:
    """
    Compute the Euclidean distance between two 1D arrays.

    The Euclidean distance is defined as:
        d(a, b) = sqrt(Σ_i (a_i - b_i)^2)

    Parameters
    ----------
    a : array_like
        First input vector (1D NumPy array, list, or tuple of numeric values).
    b : array_like
        Second input vector (1D NumPy array, list, or tuple of numeric values).

    Returns
    -------
    float
        The Euclidean distance between vectors `a` and `b`.

    Raises
    ------
    TypeError
        If `a` or `b` contains non-numeric elements.
    ValueError
        If `a` or `b` is not 1D or if shapes differ.

    Examples
    --------
    >>> import numpy as np
    >>> euclidean_distance(np.array([0, 0]), np.array([3, 4]))
    5.0
    >>> euclidean_distance([1, 2, 3], [1, 2, 3])
    0.0
    """
    a_arr, b_arr = _validate_arrays(a, b)
    return float(np.linalg.norm(a_arr - b_arr))


def manhattan_distance(a, b) -> float:
    """
    Compute the Manhattan (L1) distance between two 1D arrays.

    The Manhattan distance is defined as:
        d(a, b) = Σ_i |a_i - b_i|

    Parameters
    ----------
    a : array_like
        First input vector (1D NumPy array, list, or tuple of numeric values).
    b : array_like
        Second input vector (1D NumPy array, list, or tuple of numeric values).

    Returns
    -------
    float
        The Manhattan distance between vectors `a` and `b`.

    Raises
    ------
    TypeError
        If `a` or `b` contains non-numeric elements.
    ValueError
        If `a` or `b` is not 1D or if shapes differ.

    Examples
    --------
    >>> import numpy as np
    >>> manhattan_distance(np.array([1, 2, 3]), np.array([4, 0, 3]))
    5.0
    >>> manhattan_distance([0, 0], [0, 0])
    0.0
    """
    a_arr, b_arr = _validate_arrays(a, b)
    return float(np.sum(np.abs(a_arr - b_arr)))

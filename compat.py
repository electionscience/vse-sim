"""Small compatibility helpers shared across the simulation modules."""

from numbers import Number

from numpy import ceil as numpy_ceil
from numpy import floor as numpy_floor
from numpy import mean as numpy_mean
from numpy import median as numpy_median
from numpy import sqrt as numpy_sqrt
from numpy import std as numpy_std


def as_builtin_scalar(value):
    """Convert NumPy scalar values to their Python built-in equivalents."""
    try:
        return value.item()
    except (AttributeError, ValueError):
        return value


def ceil(*args, **kwargs):
    return as_builtin_scalar(numpy_ceil(*args, **kwargs))


def floor(*args, **kwargs):
    return as_builtin_scalar(numpy_floor(*args, **kwargs))


def mean(*args, **kwargs):
    return as_builtin_scalar(numpy_mean(*args, **kwargs))


def median(*args, **kwargs):
    return as_builtin_scalar(numpy_median(*args, **kwargs))


def sqrt(*args, **kwargs):
    return as_builtin_scalar(numpy_sqrt(*args, **kwargs))


def std(*args, **kwargs):
    return as_builtin_scalar(numpy_std(*args, **kwargs))


def isnum(x):
    """Return whether ``x`` is an instance of a numeric type."""
    return isinstance(x, Number)

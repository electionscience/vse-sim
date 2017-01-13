def isnum(x):
    """Test whether an object is an instance of a built-in numeric type."""
    for T in int, float, complex:
        if isinstance(x, T):
            return 1
    return 0

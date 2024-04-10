import pytest

def max_of_three(a, b, c):
    """
    This function returns the maximum of three numbers.
    """
    if a >= b and a >= c:
        return a
    elif b >= a and b >= c:
        return b
    else:
        return c



# test for the method above
# those tests cover each possible branch just once, no excessive repeats
def test_max_of_three():
    assert max_of_three(1, 2, 3) == 3
    assert max_of_three(3, 2, 1) == 3
    assert max_of_three(1, 3, 2) == 3
    assert max_of_three(2, 1, 3) == 3
    assert max_of_three(3, 1, 2) == 3
    assert max_of_three(2, 3, 1) == 3


# test for the method above
# those tests cover each possible branch just once, no excessive repeats

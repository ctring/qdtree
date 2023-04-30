import pytest
from qdtree.range import Range
from qdtree.dictionary import Dictionary


def test_range_overlaps():
    range_inf = Range()
    range0 = Range(0, 20, open_left=False, open_right=False)  # [0, 20]
    range1 = Range(10, 30, open_left=False, open_right=False)  # [10, 30]
    range2 = Range(20, 30, open_left=True, open_right=False)  # (20, 30]
    range3 = Range(20, 30, open_left=False, open_right=False)  # [20, 30]
    range4 = Range(10, 20, open_left=True, open_right=True)   # (10, 20)
    range5 = Range(0, 10, open_left=True, open_right=True)    # (0, 10]

    assert range_inf.overlaps(range0)
    assert range_inf.overlaps(range1)
    assert range_inf.overlaps(range2)
    assert range_inf.overlaps(range3)
    assert range0.overlaps(range_inf)
    assert range1.overlaps(range_inf)
    assert range2.overlaps(range_inf)
    assert range3.overlaps(range_inf)

    assert range0.overlaps(range1)
    assert range1.overlaps(range0)

    assert not range0.overlaps(range2)
    assert not range2.overlaps(range0)

    assert range0.overlaps(range3)
    assert range3.overlaps(range0)

    assert not range2.overlaps(range4)
    assert not range4.overlaps(range2)

    assert not range2.overlaps(range5)
    assert not range5.overlaps(range2)

    assert range0.overlaps(range0)
    assert range1.overlaps(range1)
    assert range2.overlaps(range2)
    assert range3.overlaps(range3)
    assert range4.overlaps(range4)
    assert range5.overlaps(range5)

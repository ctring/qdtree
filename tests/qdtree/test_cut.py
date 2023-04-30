import pandas as pd
import pytest
from qdtree import CutRepository, Schema
from qdtree.range import RangeWithDict, Range


@pytest.fixture
def repo() -> CutRepository:
    schema: Schema = {"count": "int", "measurement": "float"}

    builder = CutRepository.Builder(schema)

    builder.add("count", ">", "10")
    builder.add("count", "<", "20")
    builder.add("count", ">=", "30")
    builder.add("count", "<=", "40")

    builder.add("measurement", ">", "1.1")
    builder.add("measurement", "<", "2.2")
    builder.add("measurement", ">=", "3.3")
    builder.add("measurement", "<=", "4.4")

    return builder.build()


def test_cut_repository(repo: CutRepository):
    assert len(repo) == 8

    assert str(repo.get("count", ">", "10")) == "count > 10"
    assert str(repo.get("count", "<", "20")) == "count < 20"
    assert str(repo.get("count", ">=", "30")) == "count >= 30"
    assert str(repo.get("count", "<=", "40")) == "count <= 40"

    assert str(repo.get("measurement", ">", "1.1")) == "measurement > 1.1"
    assert str(repo.get("measurement", "<", "2.2")) == "measurement < 2.2"
    assert str(repo.get("measurement", ">=", "3.3")) == "measurement >= 3.3"
    assert str(repo.get("measurement", "<=", "4.4")) == "measurement <= 4.4"


def test_cut_eval_data(repo: CutRepository):
    row = pd.DataFrame({
        "count": [15],
        "measurement": [2.1],
    })

    assert repo.get("count", ">", "10").eval_data(row).all() == True
    assert repo.get("count", "<", "20").eval_data(row).all() == True
    assert repo.get("count", ">=", "30").eval_data(row).all() == False
    assert repo.get("count", "<=", "40").eval_data(row).all() == True

    assert repo.get("measurement", ">", "1.1").eval_data(row).all() == True
    assert repo.get("measurement", "<", "2.2").eval_data(row).all() == True
    assert repo.get("measurement", ">=", "3.3").eval_data(row).all() == False
    assert repo.get("measurement", "<=", "4.4").eval_data(row).all() == True


def test_cut_eval_range(repo: CutRepository):
    cut = repo.get("count", ">", "10")
    assert cut.eval_range(Range(10, 20, open_left=True, open_right=False))
    assert cut.eval_range(Range(0, 20, open_left=False, open_right=True))
    assert not cut.eval_range(Range(0, 10, open_left=False, open_right=True))
    assert not cut.eval_range(Range(0, 10, open_left=False, open_right=False))

    cut = repo.get("count", "<", "20")
    assert cut.eval_range(Range(10, 20, open_left=True, open_right=False))
    assert cut.eval_range(Range(10, 30, open_left=False, open_right=True))
    assert not cut.eval_range(Range(20, 30, open_left=False, open_right=True))
    assert not cut.eval_range(Range(20, 30, open_left=False, open_right=False))

    cut = repo.get("count", ">=", "30")
    assert cut.eval_range(Range(30, 40, open_left=True, open_right=False))
    assert cut.eval_range(Range(20, 40, open_left=False, open_right=True))
    assert not cut.eval_range(Range(20, 30, open_left=False, open_right=True))
    assert cut.eval_range(Range(20, 30, open_left=False, open_right=False))

    cut = repo.get("count", "<=", "40")
    assert cut.eval_range(Range(30, 40, open_left=True, open_right=False))
    assert cut.eval_range(Range(30, 50, open_left=False, open_right=True))
    assert not cut.eval_range(Range(40, 50, open_left=True, open_right=True))
    assert cut.eval_range(Range(40, 50, open_left=False, open_right=False))


def test_cut_range(repo: CutRepository):
    range0 = RangeWithDict(repo.dict)
    assert str(range0) == "(-inf, inf)"

    # In range cuts
    res = repo.get("count", ">", "10").cut_range(range0)
    assert res is not None
    range1, range2 = res
    assert str(range1) == "(10, inf)"
    assert str(range2) == "(-inf, 10]"

    res = repo.get("count", "<", "20").cut_range(range1)
    assert res is not None
    range3, range4 = res
    assert str(range3) == "(10, 20)"
    assert str(range4) == "[20, inf)"

    res = repo.get("count", ">=", "30").cut_range(range4)
    assert res is not None
    range5, range6 = res
    assert str(range5) == "[30, inf)"
    assert str(range6) == "[20, 30)"

    res = repo.get("count", "<=", "40").cut_range(range5)
    assert res is not None
    range7, range8 = res
    assert str(range7) == "[30, 40]"
    assert str(range8) == "(40, inf)"

    # Out of range cuts
    assert repo.get("count", ">", "10").cut_range(range8) == None
    assert repo.get("count", "<", "20").cut_range(range2) == None
    assert repo.get("count", ">=", "30").cut_range(range3) == None
    assert repo.get("count", "<=", "40").cut_range(range6) == None

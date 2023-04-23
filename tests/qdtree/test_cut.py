import pytest
from qdtree import CutRepository, Schema, Range


@pytest.fixture
def repo():
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


def test_cut_repository(repo):
    assert str(repo.get("count", ">", "10")) == "count > 10"
    assert str(repo.get("count", "<", "20")) == "count < 20"
    assert str(repo.get("count", ">=", "30")) == "count >= 30"
    assert str(repo.get("count", "<=", "40")) == "count <= 40"

    assert str(repo.get("measurement", ">", "1.1")) == "measurement > 1.1"
    assert str(repo.get("measurement", "<", "2.2")) == "measurement < 2.2"
    assert str(repo.get("measurement", ">=", "3.3")) == "measurement >= 3.3"
    assert str(repo.get("measurement", "<=", "4.4")) == "measurement <= 4.4"


def test_cut_evaluate(repo):
    row = {
        "count": 15,
        "measurement": 2.1,
    }

    assert repo.get("count", ">", "10").evaluate(row) == True
    assert repo.get("count", "<", "20").evaluate(row) == True
    assert repo.get("count", ">=", "30").evaluate(row) == False
    assert repo.get("count", "<=", "40").evaluate(row) == True

    assert repo.get("measurement", ">", "1.1").evaluate(row) == True
    assert repo.get("measurement", "<", "2.2").evaluate(row) == True
    assert repo.get("measurement", ">=", "3.3").evaluate(row) == False
    assert repo.get("measurement", "<=", "4.4").evaluate(row) == True


def test_cut_range(repo):
    range0 = Range(repo.dict)
    assert str(range0) == "(-inf, inf)"

    # In range cuts
    range1, range2 = repo.get("count", ">", "10").cut_range(range0)
    assert str(range1) == "(10, inf)"
    assert str(range2) == "(-inf, 10]"

    range3, range4 = repo.get("count", "<", "20").cut_range(range1)
    assert str(range3) == "(10, 20)"
    assert str(range4) == "[20, inf)"

    range5, range6 = repo.get("count", ">=", "30").cut_range(range4)
    assert str(range5) == "[30, inf)"
    assert str(range6) == "[20, 30)"

    range7, range8 = repo.get("count", "<=", "40").cut_range(range5)
    assert str(range7) == "[30, 40]"
    assert str(range8) == "(40, inf)"

    # Out of range cuts
    assert repo.get("count", ">", "10").cut_range(range8) == None
    assert repo.get("count", "<", "20").cut_range(range2) == None
    assert repo.get("count", ">=", "30").cut_range(range3) == None
    assert repo.get("count", "<=", "40").cut_range(range6) == None

import pytest
from qdtree import Workload, QdTree
from qdtree.range import RangeWithDict


@pytest.fixture
def workload():
    workload_dict = {
        "schema": {
            "x": "int",
            "y": "float",
            "z": "date",
        },
        "queries": {
            "q1": {
                "type": "expr",
                "children": ["z", ">=", "2020-01-01"],
            },
            "q2": {
                "type": "and",
                "children": [
                    {
                        "type": "expr",
                        "children": ["x", ">", "10"],
                    },
                    {
                        "type": "expr",
                        "children": ["y", "<", "1.5"],
                    },
                ],
            },
            "q3": {
                "type": "or",
                "children": [
                    {
                        "type": "and",
                        "children": [
                            {
                                "type": "expr",
                                "children": ["x", "<=", "20"],
                            },
                            {
                                "type": "expr",
                                "children": ["y", ">=", "0.5"],
                            },
                        ],
                    },
                    {
                        "type": "expr",
                        "children": ["z", "<=", "2020-02-01"],
                    },
                ],
            },
        },
    }
    return Workload(workload_dict)


@pytest.fixture
def qdtree(workload: Workload):
    repo = workload.cut_repo
    qdtree = QdTree(repo)
    root = qdtree.root
    root.cut(repo.get("x", ">", "10"))
    assert root.left is not None and root.right is not None
    root.left.cut(repo.get("y", "<", "1.5"))
    root.right.cut(repo.get("z", ">=", "2020-01-01"))
    assert root.left.left is not None and root.left.right is not None
    root.left.left.cut(repo.get("x", "<=", "20"))
    root.left.right.cut(repo.get("x", "<=", "20"))

    return qdtree


def test_print_workload(workload: Workload):
    print(workload)


def test_workload_cut_repo(workload: Workload):
    repo = workload.cut_repo

    assert len(repo) == 6

    assert str(repo.get("z", ">=", "2020-01-01")) == "z >= 1577854800"
    assert str(repo.get("x", ">", "10")) == "x > 10"
    assert str(repo.get("x", "<=", "20")) == "x <= 20"
    assert str(repo.get("y", "<", "1.5")) == "y < 1.5"
    assert str(repo.get("y", ">=", "0.5")) == "y >= 0.5"
    assert str(repo.get("z", "<=", "2020-02-01")) == "z <= 1580533200"


def test_workload_queries(workload: Workload):
    queries = workload._queries

    assert len(queries) == 3

    assert str(queries["q1"]) == "z >= 1577854800"
    assert str(queries["q2"]) == "(x > 10) and (y < 1.5)"
    assert (
        str(queries["q3"]) == "((x <= 20) and (y >= 0.5)) or (z <= 1580533200)"
    )


def test_predicate_can_skip(workload: Workload, qdtree: QdTree):
    blocks = sorted(qdtree.blocks, key=lambda b: str(b))

    assert len(blocks) == 6

    queries = workload._queries

    # Blocks:
    # [
    #     {'x': (-inf, 10], 'y': (-inf, inf), 'z': (-inf, 1577854800)},
    #     {'x': (-inf, 10], 'y': (-inf, inf), 'z': [1577854800, inf)},
    #     {'x': (10, 20], 'y': (-inf, 1.5), 'z': (-inf, inf)},
    #     {'x': (10, 20], 'y': [1.5, inf), 'z': (-inf, inf)},
    #     {'x': (20, inf), 'y': (-inf, 1.5), 'z': (-inf, inf)},
    #     {'x': (20, inf), 'y': [1.5, inf), 'z': (-inf, inf)}
    # ]
    #
    # Queries:
    # {
    #     'q1': z >= 1577854800,
    #     'q2': x > 10 and y < 1.5,
    #     'q3': (x <= 20 and y >= 0.5) or z <= 1580533200
    # }

    assert queries["q1"].can_skip(blocks[0])
    assert not queries["q1"].can_skip(blocks[1])
    assert not queries["q1"].can_skip(blocks[2])

    assert queries["q2"].can_skip(blocks[0])
    assert queries["q2"].can_skip(blocks[1])
    assert not queries["q2"].can_skip(blocks[2])
    assert queries["q2"].can_skip(blocks[3])
    assert not queries["q2"].can_skip(blocks[4])
    assert queries["q2"].can_skip(blocks[5])

    assert not queries["q3"].can_skip(blocks[0])
    assert not queries["q3"].can_skip(blocks[1])
    assert not queries["q3"].can_skip(blocks[2])
    assert not queries["q3"].can_skip(blocks[3])
    assert not queries["q3"].can_skip(blocks[4])
    assert not queries["q3"].can_skip(blocks[5])

import pandas as pd
import pytest
from qdtree import QdTree, Workload, Schema


@pytest.fixture
def workload():
    workload_dict = {
        "schema": {
            "x": "float",
            "y": "int",
        },
        "queries": {
            "q1": {
                "type": "expr",
                "children": ["x", "<", "0.5"],
            },
            "q2": {
                "type": "and",
                "children": [
                    {
                        "type": "expr",
                        "children": ["y", ">", "50"],
                    },
                    {
                        "type": "expr",
                        "children": ["x", "<=", "0.75"],
                    },
                ],
            },
            "q3": {
                "type": "or",
                "children": [
                    {
                        "type": "expr",
                        "children": ["y", ">=", "75"],
                    },
                    {
                        "type": "expr",
                        "children": ["x", ">=", "0.75"],
                    },
                ],
            },
        },
    }
    return Workload(workload_dict)


def test_qdtree_build_and_route(workload: Workload):
    repo = workload.cut_repo
    qdtree = QdTree(repo, pd.DataFrame())
    root = qdtree.root

    assert root._id == 1
    assert str(root.block["x"]) == "(-inf, inf)"
    assert str(root.block["y"]) == "(-inf, inf)"
    assert root.cut(repo.get("x", "<", "0.5")) == True
    assert root.cut_tracker._num_tried_cuts == 1
    if True:
        assert root.left is not None
        assert root.left._id == 2
        assert str(root.left.block["x"]) == "(-inf, 0.5)"
        assert str(root.left.block["y"]) == "(-inf, inf)"
        assert root.left.cut(repo.get("y", ">", "50")) == True
        assert root.left.cut_tracker._num_tried_cuts == 2
        if True:
            assert root.left.left is not None
            assert root.left.left._id == 4
            assert str(root.left.left.block["x"]) == "(-inf, 0.5)"
            assert str(root.left.left.block["y"]) == "(50, inf)"
            assert root.left.left.cut(repo.get("y", ">=", "75")) == True
            assert root.left.left.cut_tracker._num_tried_cuts == 3
            if True:
                assert root.left.left.left is not None
                assert root.left.left.left._id == 8
                assert str(root.left.left.left.block["x"]) == "(-inf, 0.5)"
                assert str(root.left.left.left.block["y"]) == "[75, inf)"
                assert root.left.left.left.cut_tracker._num_tried_cuts == 3

                assert root.left.left.right is not None
                assert root.left.left.right._id == 9
                assert str(root.left.left.right.block["x"]) == "(-inf, 0.5)"
                assert str(root.left.left.right.block["y"]) == "(50, 75)"
                assert root.left.left.right.cut(
                    repo.get("y", ">", "50")) == False
                assert root.left.left.right.cut_tracker._num_tried_cuts == 3

            assert root.left.right is not None
            assert root.left.right._id == 5
            assert str(root.left.right.block["x"]) == "(-inf, 0.5)"
            assert str(root.left.right.block["y"]) == "(-inf, 50]"
            assert root.left.right.cut(repo.get("y", ">=", "75")) == False
            assert root.left.right.cut_tracker._num_tried_cuts == 3

        assert root.right is not None
        assert root.right._id == 3
        assert str(root.right.block["x"]) == "[0.5, inf)"
        assert str(root.left.block["y"]) == "(-inf, inf)"
        assert root.right.cut(repo.get("x", "<=", "0.75")) == True
        assert root.right.cut_tracker._num_tried_cuts == 2
        if True:
            assert root.right.left is not None
            assert root.right.left._id == 6
            assert str(root.right.left.block["x"]) == "[0.5, 0.75]"
            assert str(root.right.left.block["y"]) == "(-inf, inf)"
            assert root.right.left.cut(repo.get("x", "<", "0.5")) == False
            assert root.right.left.cut_tracker._num_tried_cuts == 2

            assert root.right.right is not None
            assert root.right.right._id == 7
            assert str(root.right.right.block["x"]) == "(0.75, inf)"
            assert str(root.right.right.block["y"]) == "(-inf, inf)"
            assert root.right.right.cut(repo.get("x", "<=", "0.75")) == False
            assert root.right.right.cut_tracker._num_tried_cuts == 2

    print(qdtree)

    data = pd.DataFrame(
        [
            {"x": 0, "y": 75},
            {"x": 0.49999, "y": 60},
            {"x": 0.500001, "y": -200},
            {"x": 100000, "y": -100000},
        ]
    )
    expected_nodes = pd.Series([8, 9, 6, 7], index=data.index)
    assert (qdtree.route(data) == expected_nodes).all()


def test_qdtree_min_leaf_size(workload: Workload):
    repo = workload.cut_repo
    data = pd.DataFrame(
        [
            {"x": -1.0, "y": 0},
            {"x": 0, "y": 75},
            {"x": 0.4, "y": 60},
            {"x": 0.1, "y": -100000},
            {"x": 1.0, "y": -200},
            {"x": 2.0, "y": 100000},
        ]
    )

    qdtree = QdTree(repo, data, min_leaf_size=2)
    root = qdtree.root

    assert root._id == 1
    assert len(root) == 6
    assert str(root.block["x"]) == "(-inf, inf)"
    assert str(root.block["y"]) == "(-inf, inf)"
    assert root.cut(repo.get("x", "<", "0.5")) == True
    assert root.cut_tracker._num_tried_cuts == 1
    if True:
        assert root.left is not None
        assert root.left._id == 2
        assert len(root.left) == 4
        assert str(root.left.block["x"]) == "(-inf, 0.5)"
        assert str(root.left.block["y"]) == "(-inf, inf)"
        assert root.left.cut(repo.get("y", ">", "50")) == True
        assert root.left.cut_tracker._num_tried_cuts == 2
        if True:
            assert root.left.left is not None
            assert root.left.left._id == 4
            assert len(root.left.left) == 2
            assert str(root.left.left.block["x"]) == "(-inf, 0.5)"
            assert str(root.left.left.block["y"]) == "(50, inf)"
            assert root.left.left.cut(repo.get("y", ">=", "75")) == False
            assert root.left.left.cut_tracker._num_tried_cuts == 3

            assert root.left.right is not None
            assert root.left.right._id == 5
            assert len(root.left.right) == 2
            assert str(root.left.right.block["x"]) == "(-inf, 0.5)"
            assert str(root.left.right.block["y"]) == "(-inf, 50]"
            assert root.left.right.cut(repo.get("y", ">=", "75")) == False
            assert root.left.right.cut_tracker._num_tried_cuts == 3

        assert root.right is not None
        assert root.right._id == 3
        assert len(root.right) == 2
        assert str(root.right.block["x"]) == "[0.5, inf)"
        assert str(root.left.block["y"]) == "(-inf, inf)"
        assert root.right.cut(repo.get("x", "<=", "0.75")) == False
        assert root.right.cut_tracker._num_tried_cuts == 2

    print(qdtree)


def test_compute_skipped_records(workload: Workload):
    repo = workload.cut_repo
    data = pd.DataFrame(
        [
            {"x": -1.0, "y": 0},
            {"x": 0, "y": 75},
            {"x": 0.4, "y": 60},
            {"x": 0.1, "y": -100000},
            {"x": 1.0, "y": -200},
            {"x": 2.0, "y": 100000},
        ]
    )

    qdtree = QdTree(repo, data)
    root = qdtree.root

    root.cut(repo.get("x", "<", "0.5"))
    assert root.left is not None and root.right is not None
    root.left.cut(repo.get("y", ">", "50"))
    assert root.left.left is not None and root.left.right is not None
    # Queries
    # {
    #     'q1': x < 0.5,
    #     'q2': y > 50 and x <= 0.75,
    #     'q3': y >= 75 or x >= 0.75
    # }
    #
    # Blocks
    # [
    #     {'x': [0.5, inf), 'y': (-inf, inf)}  # root.right
    #     {'x': (-inf, 0.5), 'y': (50, inf)},  # root.left.left
    #     {'x': (-inf, 0.5), 'y': (-inf, 50]}, # root.left.right
    # ]
    qdtree.compute_skipped_records(workload)
    assert root.skipped_records == 6
    assert root.left.skipped_records == 4
    assert root.right.skipped_records == 2  # q1
    assert root.left.left.skipped_records == 0
    assert root.left.right.skipped_records == 4  # q2 and q3

import pandas as pd
import pytest
from qdtree import QdTree, CutRepository, Schema


@pytest.fixture
def repo():
    schema: Schema = {
        "x": "float",
        "y": "int",
    }

    builder = CutRepository.Builder(schema)
    builder.add("x", "<", "0.5")
    builder.add("y", ">", "50")
    builder.add("x", "<=", "0.75")
    builder.add("y", ">=", "75")

    return builder.build()


def test_qdtree_build_and_route(repo: CutRepository):
    qdtree = QdTree(repo, pd.DataFrame())
    root = qdtree.root

    assert root._id == 1
    assert str(root.block["x"]) == "(-inf, inf)"
    assert str(root.block["y"]) == "(-inf, inf)"
    assert root.cut(repo.get("x", "<", "0.5")) == True
    if True:
        assert root.left is not None
        assert root.left._id == 2
        assert str(root.left.block["x"]) == "(-inf, 0.5)"
        assert str(root.left.block["y"]) == "(-inf, inf)"
        assert root.left.cut(repo.get("y", ">", "50")) == True
        if True:
            assert root.left.left is not None
            assert root.left.left._id == 4
            assert str(root.left.left.block["x"]) == "(-inf, 0.5)"
            assert str(root.left.left.block["y"]) == "(50, inf)"
            assert root.left.left.cut(repo.get("y", ">=", "75")) == True
            if True:
                assert root.left.left.left is not None
                assert root.left.left.left._id == 8
                assert str(root.left.left.left.block["x"]) == "(-inf, 0.5)"
                assert str(root.left.left.left.block["y"]) == "[75, inf)"

                assert root.left.left.right is not None
                assert root.left.left.right._id == 9
                assert str(root.left.left.right.block["x"]) == "(-inf, 0.5)"
                assert str(root.left.left.right.block["y"]) == "(50, 75)"
                assert root.left.left.right.cut(
                    repo.get("y", ">", "50")) == False

            assert root.left.right is not None
            assert root.left.right._id == 5
            assert str(root.left.right.block["x"]) == "(-inf, 0.5)"
            assert str(root.left.right.block["y"]) == "(-inf, 50]"
            assert root.left.right.cut(repo.get("y", ">=", "75")) == False

        assert root.right is not None
        assert root.right._id == 3
        assert str(root.right.block["x"]) == "[0.5, inf)"
        assert str(root.left.block["y"]) == "(-inf, inf)"
        assert root.right.cut(repo.get("x", "<=", "0.75")) == True
        if True:
            assert root.right.left is not None
            assert root.right.left._id == 6
            assert str(root.right.left.block["x"]) == "[0.5, 0.75]"
            assert str(root.right.left.block["y"]) == "(-inf, inf)"
            assert root.right.left.cut(repo.get("x", "<", "0.5")) == False

            assert root.right.right is not None
            assert root.right.right._id == 7
            assert str(root.right.right.block["x"]) == "(0.75, inf)"
            assert str(root.right.right.block["y"]) == "(-inf, inf)"
            assert root.right.right.cut(repo.get("x", "<=", "0.75")) == False

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


def test_qdtree_min_leaf_size(repo: CutRepository):
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
    if True:
        assert root.left is not None
        assert root.left._id == 2
        assert len(root.left) == 4
        assert str(root.left.block["x"]) == "(-inf, 0.5)"
        assert str(root.left.block["y"]) == "(-inf, inf)"
        assert root.left.cut(repo.get("y", ">", "50")) == True
        if True:
            assert root.left.left is not None
            assert root.left.left._id == 4
            assert len(root.left.left) == 2
            assert str(root.left.left.block["x"]) == "(-inf, 0.5)"
            assert str(root.left.left.block["y"]) == "(50, inf)"
            assert root.left.left.cut(repo.get("y", ">=", "75")) == False

            assert root.left.right is not None
            assert root.left.right._id == 5
            assert len(root.left.right) == 2
            assert str(root.left.right.block["x"]) == "(-inf, 0.5)"
            assert str(root.left.right.block["y"]) == "(-inf, 50]"
            assert root.left.right.cut(repo.get("y", ">=", "75")) == False

        assert root.right is not None
        assert root.right._id == 3
        assert len(root.right) == 2
        assert str(root.right.block["x"]) == "[0.5, inf)"
        assert str(root.left.block["y"]) == "(-inf, inf)"
        assert root.right.cut(repo.get("x", "<=", "0.75")) == False

    print(qdtree)

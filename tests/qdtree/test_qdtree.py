from qdtree import QdTree, CutRepository, Schema, Range


def test_qdtree():
    schema: Schema = {
        "x": "float",
        "y": "int",
    }

    builder = CutRepository.Builder(schema)
    builder.add("x", "<", "0.5")
    builder.add("y", ">=", "10")
    builder.add("x", ">", "40")
    builder.add("y", "<=", "50")
    builder.add("y", ">", "8")

    repo = builder.build()

    ranges = {
        "x": Range(repo.dict),
        "y": Range(repo.dict),
    }

    qdtree = QdTree(ranges)
    assert qdtree.root.cut(repo.get("x", "<", "0.5")) == True
    assert qdtree.root.left is not None
    assert qdtree.root.left.cut(repo.get("y", ">=", "10")) == True
    assert qdtree.root.right is not None
    assert qdtree.root.right.cut(repo.get("y", "<=", "50")) == True
    assert qdtree.root.left.right is not None
    assert qdtree.root.left.right.cut(repo.get("x", ">", "40")) == False
    assert qdtree.root.left.right.cut(repo.get("y", ">", "8")) == True

    assert qdtree.route_tuple({"x": 0.2, "y": 10}) == 4
    assert qdtree.route_tuple({"x": 0.2, "y": 9}) == 10
    assert qdtree.route_tuple({"x": 0.8, "y": 10}) == 6

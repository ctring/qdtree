import pprint

from typing import Dict, Literal, List
from qdtree.cut import CutRepository, Cut, Operator
from qdtree.schema import Schema
from qdtree.range import Block


class Workload:
    __slots__ = ["_schema", "_cut_repo", "_queries"]

    _schema: Schema
    _cut_repo: CutRepository
    _queries: Dict[str, "Predicate"]

    @staticmethod
    def _extract_cuts(pred, repo_builder):
        typ = pred["type"]
        if typ in ["and", "or"]:
            for child in pred["children"]:
                Workload._extract_cuts(child, repo_builder)
        elif typ == "expr":
            children = pred["children"]
            repo_builder.add(children[0], children[1], children[2])
        else:
            raise ValueError(f"Invalid predicate type: {pred}")

    def __init__(self, workload_dict: Dict[str, object]):
        assert "schema" in workload_dict
        assert isinstance(workload_dict["schema"], dict)

        self._schema = workload_dict["schema"]

        assert "queries" in workload_dict
        assert isinstance(workload_dict["queries"], dict)

        # Extract the cuts from the workload and build a cut repository
        repo_builder = CutRepository.Builder(self._schema)
        for pred in workload_dict["queries"].values():
            Workload._extract_cuts(pred, repo_builder)
        self._cut_repo = repo_builder.build()

        # Build a predicate tree for each query
        self._queries = {}
        for query_id, pred in workload_dict["queries"].items():
            self._queries[query_id] = Predicate.from_dict(pred, self._cut_repo)

    def __str__(self):
        return pprint.pformat(
            {query_id: str(pred) for query_id, pred in self._queries.items()}
        )

    def __repr__(self):
        return f"Workload({repr(self._schema)}, {repr(self._queries)})"

    def __getitem__(self, query_id: str) -> "Predicate":
        return self._queries[query_id]

    @property
    def cut_repo(self):
        return self._cut_repo

    @property
    def schema(self):
        return self._schema

    def num_queries_can_skip(self, block: Block) -> int:
        return sum(1 for pred in self._queries.values() if pred.can_skip(block))


class Predicate:
    @staticmethod
    def from_dict(pred, repo: CutRepository) -> "Predicate":
        typ = pred["type"]
        if typ in ["and", "or"]:
            children = [Predicate.from_dict(child, repo)
                        for child in pred["children"]]
            return BoolOp(typ, children)
        elif typ == "expr":
            attr1, op, attr2 = pred["children"]
            return CutExpr(repo.get(attr1, op, attr2))  # type: ignore
        else:
            raise ValueError(f"Invalid predicate type: {pred}")

    def can_skip(self, _: Block) -> bool:
        raise NotImplementedError()


class BoolOp(Predicate):
    def __init__(self, op: Literal["and", "or"], children: List[Predicate]):
        self.op = op
        self.children = children

    def __str__(self):
        return f" {self.op} ".join(f"({str(child)})" for child in self.children)

    def __repr__(self):
        return f" {self.op}(" + ", ".join(repr(child) for child in self.children) + ")"

    def can_skip(self, block: Block) -> bool:
        children_eval = [child.can_skip(block) for child in self.children]
        if self.op == "and":
            # If any child skip the block, then the block can be skipped
            return any(children_eval)
        elif self.op == "or":
            # If all children skip the block, then the block can be skipped
            return all(children_eval)
        else:
            raise ValueError(f"Invalid operator: {self.op}")


class CutExpr(Predicate):
    def __init__(self, cut: Cut):
        self.cut = cut

    def __str__(self):
        return str(self.cut)

    def __repr__(self):
        return repr(self.cut)

    def can_skip(self, block: Block) -> bool:
        attr = self.cut.attr1
        return not self.cut.overlaps_range(block[attr])

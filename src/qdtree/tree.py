import copy
import pprint
import numpy as np
import pandas as pd

from typing import List, NamedTuple, Optional, Set, Tuple

from qdtree.cut import Cut, CutRepository
from qdtree.range import RangeWithDict, Block
from qdtree.workload import Workload


class QdTreeContext(NamedTuple):
    attributes: List[str]
    min_leaf_size: int
    leaves: Set["QdTreeNode"]


class QdTreeNode:
    __slots__ = [
        "_context",
        "_id",
        "_block",
        "_data",
        "_cut",
        "_left",
        "_right",
        "_skipped_records",
        "_cut_tracker",
    ]

    _cut: Optional[Cut]
    _left: Optional["QdTreeNode"]
    _right: Optional["QdTreeNode"]
    _skipped_records: int

    def __init__(
        self,
        context: QdTreeContext,
        id: int,
        block: Block,
        data: pd.DataFrame,
    ):
        self._context = context
        self._id = id
        self._block = block
        self._data = data
        self._cut = None
        self._left = None
        self._right = None
        self._skipped_records = 0

        # A new node is always a leaf
        self._context.leaves.add(self)

    def __str__(self):
        return pprint.pformat(self.__dict__(with_children=True), sort_dicts=False)

    def __repr__(self):
        return pprint.pformat(self.__dict__(with_children=False), sort_dicts=False)

    def __len__(self):
        return len(self._data)

    def __dict__(self, with_children):
        d = {
            "id": self._id,
            "cut": self._cut,
            "size": len(self),
            "block": self._block,
            "skipped_records": self._skipped_records,
        }
        if with_children:
            d["left"] = (
                self._left.__dict__(with_children) if self._left is not None else None
            )
            d["right"] = (
                self._right.__dict__(with_children) if self._right is not None else None
            )
        return d

    @property
    def id(self) -> int:
        return self._id

    @property
    def left(self) -> Optional["QdTreeNode"]:
        return self._left

    @property
    def right(self) -> Optional["QdTreeNode"]:
        return self._right

    @property
    def block(self) -> Block:
        return self._block

    @property
    def skipped_records(self) -> int:
        return self._skipped_records

    @property
    def encoding(self) -> np.ndarray:
        return np.concatenate(
            [self._block[attr].encoding for attr in self._context.attributes]
        )

    @property
    def encoding_space(self) -> Tuple[np.ndarray, np.ndarray]:
        node_low = []
        node_high = []
        for attr in self._context.attributes:
            low, high = self._block[attr].encoding_space
            node_low.append(low)
            node_high.append(high)
        return np.concatenate(node_low), np.concatenate(node_high)

    def cut(self, cut: Cut) -> bool:
        """Cut the tree at this node.

        Returns True if the cut was successful, False otherwise.
        """
        if self._cut is not None:
            raise RuntimeError("Cut already exists")

        assert cut.attr1 in self._block, f"Attribute {cut.attr1} not in block"

        new_block = cut.cut_range(self._block[cut.attr1])
        if new_block is None:
            return False

        pos_data = neg_data = self._data
        if len(self._data) > 0:
            cut_eval_true = cut.eval_data(self._data)
            pos_data = self._data[cut_eval_true]
            neg_data = self._data[~cut_eval_true]

        if (
            len(pos_data) < self._context.min_leaf_size
            or len(neg_data) < self._context.min_leaf_size
        ):
            return False

        true_range, false_range = new_block

        self._left = QdTreeNode(
            self._context,
            self.id * 2,
            {**self._block, cut.attr1: true_range},
            pos_data,
        )

        self._right = QdTreeNode(
            self._context,
            self.id * 2 + 1,
            {**self._block, cut.attr1: false_range},
            neg_data,
        )
        self._cut = cut

        # A cut node is no longer a leaf
        self._context.leaves.remove(self)

        return True

    def route(self, rows: pd.DataFrame) -> pd.Series:
        """Route a tuple to a leaf node.

        Returns the id of the leaf node.
        """
        if len(rows) == 0:
            return pd.Series(dtype=int)

        if self._cut is None:
            return pd.Series(self._id, index=rows.index)

        assert self._left is not None and self._right is not None

        results = pd.Series(index=rows.index, dtype=int)
        cut_eval_true = self._cut.eval_data(rows)
        results[cut_eval_true] = self._left.route(rows[cut_eval_true])
        results[~cut_eval_true] = self._right.route(rows[~cut_eval_true])

        return results

    def compute_skipped_records(self):
        if self._cut is None:
            return

        assert self._left is not None and self._right is not None

        self._left.compute_skipped_records()
        self._right.compute_skipped_records()

        self._skipped_records = (
            self._left._skipped_records + self._right._skipped_records
        )


class QdTree:
    __slots__ = ["_context", "_root"]

    _context: QdTreeContext
    _root: QdTreeNode

    def __init__(
        self,
        repo: CutRepository,
        data: pd.DataFrame = pd.DataFrame(),
        min_leaf_size: int = 0,
    ):
        self._context = QdTreeContext(
            attributes=list(repo.schema.keys()),
            min_leaf_size=min_leaf_size,
            leaves=set(),
        )
        block = {attr: RangeWithDict(repo.dict) for attr in repo.schema}
        self._root = QdTreeNode(self._context, 1, block, data)

    def __str__(self):
        return f"{self._root}"

    def __repr__(self):
        return self.__str__()

    @property
    def root(self) -> QdTreeNode:
        return self._root

    @property
    def blocks(self) -> List[Block]:
        return [node.block for node in self._context.leaves]

    def route(self, rows: pd.DataFrame) -> pd.Series:
        """Route a tuple to a leaf node.

        Returns the id of the leaf node.
        """
        return self._root.route(rows)

    def compute_skipped_records(self, workload: Workload):
        """Compute the number of skipped records for each node in the tree."""
        # Populate the number of skipped records for the leaves
        for node in self._context.leaves:
            node._skipped_records = workload.num_queries_can_skip(node.block) * len(
                node
            )

        # Recursively compute the results for the inner nodes
        self._root.compute_skipped_records()

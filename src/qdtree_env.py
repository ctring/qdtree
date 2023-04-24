import gymnasium as gym
import numpy as np
import pandas as pd

from collections import deque
from typing import Deque, Optional
from gymnasium.spaces import Discrete, Box
from ray.rllib.env.env_context import EnvContext

from qdtree import *


class QdTreeEnv(gym.Env[np.ndarray, int]):
    """Environment for building a qd-tree."""

    cut_repo: CutRepository
    data: pd.DataFrame
    min_leaf_size: int
    qd_tree: QdTree

    cur_node: Optional[QdTreeNode]
    queue: Deque[QdTreeNode]

    def __init__(self, config: EnvContext):
        self.cut_repo = config["cut_repo"]
        self.data = config["data"]
        self.min_leaf_size = config["min_leaf_size"]
        self.qd_tree = QdTree(self.cut_repo, self.data, self.min_leaf_size)

        self.action_space = Discrete(len(self.cut_repo))

        low, high = self.qd_tree.root.encode_space()
        self.observation_space = Box(low=low, high=high, dtype=np.int32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        self.queue = deque()
        self.qd_tree = QdTree(self.cut_repo, self.data, self.min_leaf_size)
        self.cur_node = self.qd_tree.root

        return self.cur_node.encode(), {}

    def step(self, action: int):
        assert self.cur_node is not None

        cut = self.cut_repo[action]

        if self.cur_node.cut(cut):
            assert self.cur_node.left is not None
            self.queue.append(self.cur_node.left)

            assert self.cur_node.right is not None
            self.queue.append(self.cur_node.right)

        done = True
        if len(self.queue) > 0:
            self.cur_node = self.queue.popleft()
            done = False

        return (
            self.cur_node.encode() if self.cur_node is not None else None,
            0,  # reward
            done,
            False,  # truncated
            {},
        )

    def print_tree(self):
        print(self.qd_tree)


if __name__ == "__main__":
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

    config = EnvContext(
        {
            "cut_repo": repo,
            "data": pd.DataFrame(
                [
                    {"x": 0.2, "y": 10},
                    {"x": 0.4, "y": 20},
                    {"x": 0.6, "y": 30},
                    {"x": 0.8, "y": 40},
                ]
            ),
            "min_leaf_size": 2,
        },
        0,
    )

    env = QdTreeEnv(config)

    init, _ = env.reset()
    env.print_tree()

    env.step(0)
    env.print_tree()

    env.step(1)
    env.print_tree()

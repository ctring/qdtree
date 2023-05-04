import gymnasium as gym
import numpy as np
import pandas as pd

from collections import deque
from typing import Deque, List, Optional
from gymnasium.spaces import Discrete, Box
from ray.rllib.env.env_context import EnvContext

from qdtree import *


class QdTreeEnv(gym.Env[np.ndarray, int]):
    """Environment for building a qd-tree."""

    workload: Workload
    data: pd.DataFrame
    min_leaf_size: int

    qd_tree: QdTree
    cur_node: QdTreeNode
    queue: Deque[QdTreeNode]
    node_history: List[Optional[QdTreeNode]]
    done: bool

    def __init__(self, config: EnvContext):
        # Extract the configuration.
        self.workload = config["workload"]
        self.data = config["data"]
        self.min_leaf_size = config["min_leaf_size"]
        self.qd_tree = QdTree(self.workload.cut_repo,
                              self.data, self.min_leaf_size)

        # Set up the action and observation spaces.
        self.action_space = Discrete(len(self.workload.cut_repo))
        low, high = self.qd_tree.root.encoding_space
        self.observation_space = Box(low=low, high=high, dtype=np.int32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.qd_tree = QdTree(self.workload.cut_repo,
                              self.data, self.min_leaf_size)
        self.cur_node = self.qd_tree.root
        self.queue = deque()
        self.node_history = [self.cur_node]
        self.done = False

        return self.cur_node.encoding, {}

    def step(self, action: int):
        if self.done:
            return self.cur_node.encoding, 0, True, False, {}

        cut = self.workload.cut_repo[action]
        pop_next_node = False

        # Make a cut and add the new nodes to the queue for later visits.
        if self.cur_node.cut(cut):
            assert self.cur_node.left is not None
            self.queue.append(self.cur_node.left)
            assert self.cur_node.right is not None
            self.queue.append(self.cur_node.right)
            pop_next_node = True
        # If the cut is not possible, then check if we are done with the current
        # node yet.
        elif self.cur_node.cut_tracker.is_done():
            pop_next_node = True

        if pop_next_node:
            if len(self.queue) > 0:
                self.cur_node = self.queue.popleft()
                self.node_history.append(self.cur_node)
            else:
                # No more node to explore. We don't return self.done immediately
                # from here, but just set this variable to True and return it in
                # the next step() call. This is because rllib does not register
                # the info of the last step, so we add an additional step at the
                # end to return the info properly.
                self.done = True
        else:
            self.node_history.append(None)

        obs = self.cur_node.encoding
        info = {}
        reward = 0
        if self.done:
            info["rewards"] = self._compute_rewards()
            reward = info["rewards"][0]  # type: ignore

        # Note that we don't return self.done here for the terminated entry
        # The reward returned here is only used for metrics summary and not
        # for training the policy. The actual rewards are returned as part of
        # the info dict.
        return obs, reward, False, False, info

    def render(self):
        return str(self.qd_tree)

    def _compute_rewards(self):
        """Compute the rewards for each node in the tree."""
        self.qd_tree.compute_skipped_records(self.workload)
        rewards = np.array([
            (
                node.skipped_records / (len(node) * len(self.workload))
                if node else 0
            )
            for node in self.node_history
        ] + [0])   # Append a reward for the last action, which is only a dummy
        return rewards

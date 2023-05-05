from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch
from ray.rllib.policy.sample_batch import SampleBatch

import torch


class QdTreePolicy(PPOTorchPolicy):
    def postprocess_trajectory(
        self, sample_batch: SampleBatch, other_agent_batches=None, episode=None
    ):
        info = sample_batch[SampleBatch.INFOS][-1]
        if isinstance(info, dict):
            if "rewards" not in info:
                raise RuntimeError(
                    'Cannot find "rewards" in the info dict. This means the episode has been truncated.'
                    "Set batch_mode of the rollouts config to complete_episodes to avoid this error."
                )
            assert len(sample_batch[SampleBatch.REWARDS]) == len(info["rewards"])
            sample_batch[SampleBatch.REWARDS] = info["rewards"]

        # Copied from postprocess_trajectory of PPOTorchPolicy
        assert torch is not None
        with torch.no_grad():
            return compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )

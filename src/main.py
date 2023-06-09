import argparse
import json
import pandas as pd
import os

import ray
from ray import air, tune
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls

from env import QdTreeEnv
from policy import QdTreePolicy
from qdtree import Workload
from qdtree.schema import ensure_data_schema

parser = argparse.ArgumentParser()
parser.add_argument("--workload", type=str, required=True, help="Workload json file.")
parser.add_argument(
    "--data", type=str, required=True, help="Data file. Can be csv or parquet."
)
parser.add_argument(
    "--sample-pct", type=float, default=1.0, help="Percentage of data to sample."
)
parser.add_argument(
    "--min-leaf-size", type=int, required=True, help="Minimum leaf size for QdTree."
)
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=50, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=0.1, help="Reward at which we stop training."
)
parser.add_argument(
    "--no-tune",
    action="store_true",
    help="Run without Tune using a manual train loop instead. In this case,"
    "use PPO without grid search and no TensorBoard.",
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    with open(args.workload, "r") as f:
        workload = Workload(json.load(f))

    print("Loading data...")
    if args.data.endswith(".csv"):
        data = pd.read_csv(args.data)
    elif args.data.endswith(".parquet"):
        data = pd.read_parquet(args.data)
    else:
        raise ValueError("Data file must be csv or parquet.")
    print(f"Loaded {len(data)} rows.")

    if args.sample_pct < 1.0:
        data = data.sample(frac=args.sample_pct)
        print(f"Sampled {len(data)} rows.")

    data.columns = data.columns.str.lower()
    data = ensure_data_schema(data, workload.schema)

    ray.init(local_mode=args.local_mode)

    config = (
        get_trainable_cls(args.run)
        .get_default_config()
        .environment(
            QdTreeEnv,
            env_config={
                "workload": workload,
                "data": data,
                "min_leaf_size": args.min_leaf_size,
            },
        )
        .framework("torch")
        .rollouts(num_rollout_workers=1, batch_mode="complete_episodes")
        .training(
            model={
                "fcnet_hiddens": [512, 512],
                "fcnet_activation": "relu",
                # "vf_share_layers": True,
            }
        )
        .multi_agent(
            policies={DEFAULT_POLICY_ID: PolicySpec(policy_class=QdTreePolicy)}
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    algo = config.build()

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    if args.no_tune:
        # manual training with train loop using PPO and fixed learning rate
        if args.run != "PPO":
            raise ValueError("Only support --run PPO with --no-tune.")
        print("Running manual train loop without Ray Tune.")
        # use fixed learning rate instead of grid search (needs tune)
        config.lr = 1e-3
        algo = config.build()
        # run manual training loop and print results after each iteration
        for _ in range(args.stop_iters):
            result = algo.train()
            print(pretty_print(result))
            # stop training of the target train steps or reward are reached
            if (
                result["timesteps_total"] >= args.stop_timesteps
                or result["episode_reward_mean"] >= args.stop_reward
            ):
                break
        algo.stop()
    else:
        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        tuner = tune.Tuner(
            args.run,
            param_space=config.to_dict(),
            run_config=air.RunConfig(
                local_dir="./results",
                stop=stop,
                checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True),
            ),
        )
        results = tuner.fit()

        if args.as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, args.stop_reward)

    ray.shutdown()

import json
from pathlib import Path
from typing import cast

import numpy as np
import tinker

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.rollout_logging import (
    serialize_rollout_summaries,
    write_rollout_summaries_jsonl,
)
from tinker_cookbook.rl.types import Logs, Metrics, Trajectory, TrajectoryGroup, Transition


def test_write_rollout_summaries_jsonl_handles_numpy_scalars(tmp_path: Path):
    transition = Transition(
        ob=tinker.ModelInput.from_ints([101, 102, 103, 104, 105]),
        ac=TokensWithLogprobs(tokens=[1, 2, 3], maybe_logprobs=[-0.1, -0.2, -0.3]),
        reward=cast(float, np.float32(0.25)),
        episode_done=True,
        metrics=cast(Metrics, {"score": np.float32(1.5)}),
        logs=cast(Logs, {"rank": np.int64(2)}),
    )
    trajectory = Trajectory(transitions=[transition], final_ob=tinker.ModelInput.from_ints([1] * 8))
    trajectory_group = TrajectoryGroup(
        trajectories_G=[trajectory],
        final_rewards_G=[cast(float, np.float32(0.75))],
        metrics_G=[cast(Metrics, {"traj_metric": np.float32(3.0)})],
    )
    output_path = tmp_path / "rollouts.jsonl"

    write_rollout_summaries_jsonl(
        output_path,
        split="train",
        iteration=1,
        trajectory_groups_P=[trajectory_group],
        taglist_P=[["unit-test"]],
        sampling_client_steps_P=[7],
    )

    record = json.loads(output_path.read_text().strip())
    assert record["iteration"] == 1
    assert record["sampling_client_step"] == 7
    assert record["total_reward"] == 1.0
    assert record["final_reward"] == 0.75
    assert record["steps"][0]["reward"] == 0.25
    assert record["steps"][0]["metrics"]["score"] == 1.5
    assert record["steps"][0]["logs"]["rank"] == 2


def _make_trajectory_group(ac: TokensWithLogprobs) -> TrajectoryGroup:
    transition = Transition(
        ob=tinker.ModelInput.from_ints([101, 102, 103]),
        ac=ac,
        reward=1.0,
        episode_done=True,
    )
    trajectory = Trajectory(transitions=[transition], final_ob=tinker.ModelInput.from_ints([1]))
    return TrajectoryGroup(
        trajectories_G=[trajectory],
        final_rewards_G=[0.0],
        metrics_G=[{}],
    )


def test_serialize_rollout_summaries_includes_sampler_provenance():
    """Step records carry policy_version/sample_retries/sample_retry_wait_s and schema v2."""
    trajectory_group = _make_trajectory_group(
        TokensWithLogprobs(
            tokens=[1, 2],
            maybe_logprobs=[-0.1, -0.2],
            policy_version=12,
            sample_retries=1,
            sample_retry_wait_s=0.5,
        )
    )

    records = serialize_rollout_summaries(
        split="train",
        iteration=3,
        trajectory_groups_P=[trajectory_group],
        taglist_P=[["unit-test"]],
    )

    assert len(records) == 1
    record = records[0]
    assert record["schema_version"] == 2
    step = record["steps"][0]
    assert step["policy_version"] == 12
    assert step["sample_retries"] == 1
    assert step["sample_retry_wait_s"] == 0.5


def test_serialize_rollout_summaries_none_provenance_round_trips():
    """Default (None) provenance fields survive a JSON round-trip as nulls."""
    trajectory_group = _make_trajectory_group(
        TokensWithLogprobs(tokens=[1, 2], maybe_logprobs=None)
    )

    records = serialize_rollout_summaries(
        split="train",
        iteration=0,
        trajectory_groups_P=[trajectory_group],
        taglist_P=[["unit-test"]],
    )

    round_tripped = json.loads(json.dumps(records[0]))
    assert round_tripped["schema_version"] == 2
    step = round_tripped["steps"][0]
    assert step["policy_version"] is None
    assert step["sample_retries"] is None
    assert step["sample_retry_wait_s"] is None

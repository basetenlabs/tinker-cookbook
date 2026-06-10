"""Tests for extended KL / policy-version metrics in tinker_cookbook.rl.metrics."""

import json
import math
from typing import Any

import pytest
import tinker
import torch

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.metrics import (
    compute_kl_sample_train,
    compute_kl_sample_train_extended,
    compute_policy_version_metrics,
    read_trainer_policy_version,
)
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition

# --- Helpers ---


def _make_datum(mask: list[float], sampling_logprobs: list[float]) -> tinker.Datum:
    n = len(mask)
    assert len(sampling_logprobs) == n
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(list(range(1, n + 1))),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(list(range(2, n + 2)))),
            "logprobs": tinker.TensorData.from_torch(torch.tensor(sampling_logprobs)),
            "advantages": tinker.TensorData.from_torch(torch.zeros(n)),
            "mask": tinker.TensorData.from_torch(torch.tensor(mask)),
        },
    )


def _quantile(values: list[float], q: float) -> float:
    """Linear-interpolation quantile matching torch.quantile's default."""
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    return ordered[lo] + (pos - lo) * (ordered[hi] - ordered[lo])


def _make_transition(
    policy_version: int | None = None,
    sample_retries: int | None = None,
    sample_retry_wait_s: float | None = None,
) -> Transition:
    return Transition(
        ob=tinker.ModelInput.from_ints([1, 2, 3]),
        ac=TokensWithLogprobs(
            tokens=[4, 5],
            maybe_logprobs=[-0.1, -0.2],
            policy_version=policy_version,
            sample_retries=sample_retries,
            sample_retry_wait_s=sample_retry_wait_s,
        ),
        reward=0.0,
        episode_done=True,
    )


def _make_group(transitions_per_trajectory: list[list[Transition]]) -> TrajectoryGroup:
    trajectories = [
        Trajectory(transitions=transitions, final_ob=tinker.ModelInput.from_ints([]))
        for transitions in transitions_per_trajectory
    ]
    return TrajectoryGroup(
        trajectories_G=trajectories,
        final_rewards_G=[0.0] * len(trajectories),
        metrics_G=[{} for _ in trajectories],
    )


# --- compute_kl_sample_train_extended ---
#
# Fixture batch (diff = sampling_lp - training_lp on action tokens):
#   datum 0: mask [0,1,1,0,1] -> two action runs ([1,2] and [4]); diffs [0.5, -0.5, 0.5]
#   datum 1: all-zero mask -> skipped from extended stats
#   datum 2: mask [0,1]; diff [0.8]
# flat diffs = [0.5, -0.5, 0.5, 0.8]

_DATUM0_DIFFS = [0.5, -0.5, 0.5]
_DATUM2_DIFFS = [0.8]
_FLAT_DIFFS = _DATUM0_DIFFS + _DATUM2_DIFFS


def _make_batch() -> tuple[list[tinker.Datum], list[torch.Tensor]]:
    data_D = [
        _make_datum(
            mask=[0.0, 1.0, 1.0, 0.0, 1.0],
            sampling_logprobs=[0.0, -1.0, -2.0, 0.0, -3.0],
        ),
        _make_datum(mask=[0.0, 0.0], sampling_logprobs=[-0.5, -0.5]),
        _make_datum(mask=[0.0, 1.0], sampling_logprobs=[-0.2, -1.0]),
    ]
    training_logprobs_D = [
        torch.tensor([0.0, -1.5, -1.5, 0.0, -3.5]),
        torch.tensor([-0.5, -0.5]),
        torch.tensor([-0.2, -1.8]),
    ]
    return data_D, training_logprobs_D


class TestComputeKlSampleTrainExtended:
    def test_includes_base_metrics(self):
        data_D, training_logprobs_D = _make_batch()
        details = compute_kl_sample_train_extended(data_D, training_logprobs_D)
        base = compute_kl_sample_train(data_D, training_logprobs_D)
        for key, value in base.items():
            assert details.metrics[key] == pytest.approx(value)
        assert details.metrics["optim/kl_sample_train_v1"] == pytest.approx(
            sum(_FLAT_DIFFS) / len(_FLAT_DIFFS)
        )

    def test_logprob_diff_percentiles(self):
        data_D, training_logprobs_D = _make_batch()
        metrics = compute_kl_sample_train_extended(data_D, training_logprobs_D).metrics
        assert metrics["optim/logprob_diff/p05"] == pytest.approx(_quantile(_FLAT_DIFFS, 0.05))
        assert metrics["optim/logprob_diff/p50"] == pytest.approx(_quantile(_FLAT_DIFFS, 0.50))
        assert metrics["optim/logprob_diff/p95"] == pytest.approx(_quantile(_FLAT_DIFFS, 0.95))
        assert metrics["optim/logprob_diff/p99"] == pytest.approx(_quantile(_FLAT_DIFFS, 0.99))
        assert metrics["optim/logprob_diff/max_abs"] == pytest.approx(0.8)
        assert metrics["optim/logprob_diff/frac_neg"] == pytest.approx(0.25)

    def test_is_ratio_stats(self):
        data_D, training_logprobs_D = _make_batch()
        metrics = compute_kl_sample_train_extended(data_D, training_logprobs_D).metrics
        ratios = [math.exp(-d) for d in _FLAT_DIFFS]
        assert metrics["optim/is_ratio/mean"] == pytest.approx(sum(ratios) / len(ratios))
        assert metrics["optim/is_ratio/p99"] == pytest.approx(_quantile(ratios, 0.99))
        assert metrics["optim/is_ratio/max"] == pytest.approx(max(ratios))
        # exp(0.5) ~ 1.65 -> nothing above 2; exp(-0.8) ~ 0.449 -> one below 0.5
        assert metrics["optim/is_ratio/frac_gt_2"] == pytest.approx(0.0)
        assert metrics["optim/is_ratio/frac_lt_half"] == pytest.approx(0.25)

    def test_kl_datum_stats(self):
        data_D, training_logprobs_D = _make_batch()
        metrics = compute_kl_sample_train_extended(data_D, training_logprobs_D).metrics
        datum_means = [
            sum(_DATUM0_DIFFS) / len(_DATUM0_DIFFS),
            sum(_DATUM2_DIFFS) / len(_DATUM2_DIFFS),
        ]
        assert metrics["optim/kl_datum/p50"] == pytest.approx(_quantile(datum_means, 0.50))
        assert metrics["optim/kl_datum/p95"] == pytest.approx(_quantile(datum_means, 0.95))
        assert metrics["optim/kl_datum/max"] == pytest.approx(max(datum_means))

    def test_substep_split(self):
        data_D, training_logprobs_D = _make_batch()
        metrics = compute_kl_sample_train_extended(
            data_D, training_logprobs_D, substep_ids_D=[0, 0, 1]
        ).metrics
        assert metrics["optim/kl_sample_train_v1/substep_0"] == pytest.approx(
            sum(_DATUM0_DIFFS) / len(_DATUM0_DIFFS)
        )
        assert metrics["optim/kl_sample_train_v1/substep_1"] == pytest.approx(0.8)

    def test_no_substep_keys_for_single_substep(self):
        data_D, training_logprobs_D = _make_batch()
        for substep_ids_D in (None, [0, 0, 0]):
            metrics = compute_kl_sample_train_extended(
                data_D, training_logprobs_D, substep_ids_D=substep_ids_D
            ).metrics
            assert not any(key.startswith("optim/kl_sample_train_v1/substep_") for key in metrics)

    def test_per_datum_records(self):
        data_D, training_logprobs_D = _make_batch()
        details = compute_kl_sample_train_extended(
            data_D, training_logprobs_D, substep_ids_D=[0, 0, 1]
        )
        # Empty-mask datum (index 1) is skipped
        assert [record["datum_idx"] for record in details.per_datum] == [0, 2]

        record0 = details.per_datum[0]
        assert record0["substep"] == 0
        assert record0["n_action_tokens"] == 3
        assert record0["mean_diff"] == pytest.approx(sum(_DATUM0_DIFFS) / 3)
        assert record0["p95_diff"] == pytest.approx(_quantile(_DATUM0_DIFFS, 0.95))
        assert record0["max_abs_diff"] == pytest.approx(0.5)
        assert record0["frac_neg"] == pytest.approx(1 / 3)
        ratios0 = [math.exp(-d) for d in _DATUM0_DIFFS]
        assert record0["is_ratio_mean"] == pytest.approx(sum(ratios0) / 3)
        assert record0["is_ratio_max"] == pytest.approx(max(ratios0))

        record2 = details.per_datum[1]
        assert record2["substep"] == 1
        assert record2["n_action_tokens"] == 1
        assert record2["mean_diff"] == pytest.approx(0.8)

    def test_per_turn_segmentation(self):
        data_D, training_logprobs_D = _make_batch()
        details = compute_kl_sample_train_extended(data_D, training_logprobs_D)
        # Datum 0 mask [0,1,1,0,1] -> two contiguous action runs
        turns = details.per_datum[0]["turns"]
        assert len(turns) == 2
        assert turns[0]["turn_idx"] == 0
        assert turns[0]["n_tokens"] == 2
        assert turns[0]["mean_diff"] == pytest.approx(0.0)
        assert turns[0]["first_token_diff"] == pytest.approx(0.5)
        assert turns[1]["turn_idx"] == 1
        assert turns[1]["n_tokens"] == 1
        assert turns[1]["mean_diff"] == pytest.approx(0.5)
        assert turns[1]["first_token_diff"] == pytest.approx(0.5)
        # Datum 2 mask [0,1] -> single run at the end
        turns2 = details.per_datum[1]["turns"]
        assert len(turns2) == 1
        assert turns2[0]["n_tokens"] == 1
        assert turns2[0]["first_token_diff"] == pytest.approx(0.8)

    def test_per_datum_is_json_safe(self):
        data_D, training_logprobs_D = _make_batch()
        details = compute_kl_sample_train_extended(data_D, training_logprobs_D)
        round_tripped = json.loads(json.dumps(details.per_datum))
        assert round_tripped == details.per_datum
        assert json.loads(json.dumps(details.metrics)) == details.metrics


# --- compute_policy_version_metrics ---


class TestComputePolicyVersionMetrics:
    def test_mixed_versions_and_retries(self):
        groups = [
            _make_group(
                [
                    [
                        _make_transition(
                            policy_version=3, sample_retries=2, sample_retry_wait_s=1.5
                        ),
                        _make_transition(policy_version=None),
                    ]
                ]
            ),
            _make_group(
                [[_make_transition(policy_version=5, sample_retries=1, sample_retry_wait_s=0.25)]]
            ),
        ]
        metrics = compute_policy_version_metrics(groups, trainer_version_pre=7)
        assert metrics["policy_version/sampler_min"] == 3.0
        assert metrics["policy_version/sampler_max"] == 5.0
        assert metrics["policy_version/frac_missing"] == pytest.approx(1 / 3)
        assert metrics["policy_version/trainer_pre"] == 7.0
        assert metrics["policy_version/lag"] == 2.0
        assert metrics["sample_retry/count_total"] == 3.0
        assert metrics["sample_retry/wait_total_s"] == pytest.approx(1.75)
        assert metrics["sample_retry/wait_max_s"] == pytest.approx(1.5)

    def test_all_none_yields_empty_dict(self):
        groups = [_make_group([[_make_transition(), _make_transition()]])]
        assert compute_policy_version_metrics(groups, trainer_version_pre=None) == {}

    def test_trainer_version_only(self):
        groups = [_make_group([[_make_transition()]])]
        metrics = compute_policy_version_metrics(groups, trainer_version_pre=4)
        assert metrics == {"policy_version/trainer_pre": 4.0}

    def test_versions_without_trainer(self):
        groups = [_make_group([[_make_transition(policy_version=9)]])]
        metrics = compute_policy_version_metrics(groups, trainer_version_pre=None)
        assert metrics["policy_version/sampler_min"] == 9.0
        assert metrics["policy_version/sampler_max"] == 9.0
        assert metrics["policy_version/frac_missing"] == 0.0
        assert "policy_version/trainer_pre" not in metrics
        assert "policy_version/lag" not in metrics
        assert "sample_retry/count_total" not in metrics


# --- read_trainer_policy_version ---


class _NoVersionClient:
    pass


class _RaisingClient:
    @property
    def policy_version(self) -> int:
        raise RuntimeError("HTTP GET failed")


class _IntVersionClient:
    policy_version = 42


class _NoneVersionClient:
    policy_version = None


class TestReadTrainerPolicyVersion:
    def test_missing_attribute_returns_none(self):
        client: Any = _NoVersionClient()
        assert read_trainer_policy_version(client) is None

    def test_raising_property_returns_none(self):
        client: Any = _RaisingClient()
        assert read_trainer_policy_version(client) is None

    def test_int_value_returned(self):
        client: Any = _IntVersionClient()
        assert read_trainer_policy_version(client) == 42

    def test_none_value_returns_none(self):
        client: Any = _NoneVersionClient()
        assert read_trainer_policy_version(client) is None

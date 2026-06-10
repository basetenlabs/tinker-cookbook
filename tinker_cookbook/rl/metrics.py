"""
Metrics and KL computation functions for RL training.

Contains functions for computing KL divergences, incorporating KL penalties,
and computing training metrics.
"""

import asyncio
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import tinker
import torch

from tinker_cookbook.rl.types import TrajectoryGroup
from tinker_cookbook.utils import trace
from tinker_cookbook.utils.misc_utils import safezip


def compute_kl_sample_train(
    data_D: list[tinker.Datum], training_logprobs_D: list[torch.Tensor]
) -> dict[str, float]:
    """Compute KL divergence metrics between sampling and training logprobs.

    Compares the logprobs from when trajectories were sampled against the current
    training logprobs to measure how much the policy has shifted. Computes both
    first-order (v1) and second-order (v2) KL estimates, as well as the sampling
    entropy. Only action tokens (where mask > 0) are included.

    Args:
        data_D (list[tinker.Datum]): List of datums, each containing ``logprobs``
            and ``mask`` tensors in ``loss_fn_inputs``.
        training_logprobs_D (list[torch.Tensor]): Per-token logprobs from the
            current training model, one tensor per datum.

    Returns:
        dict[str, float]: Dictionary with keys:
            - ``optim/kl_sample_train_v1``: Mean logprob difference (first-order KL estimate).
            - ``optim/kl_sample_train_v2``: Half mean squared logprob difference (second-order KL).
            - ``optim/entropy``: Estimated entropy of the sampling distribution.
    """
    all_diffs: list[torch.Tensor] = []
    all_sampling_logprobs: list[torch.Tensor] = []

    for datum, training_logprobs in safezip(data_D, training_logprobs_D):
        # Get logprobs from sampling
        sampling_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
        action_mask = datum.loss_fn_inputs["mask"].to_torch() > 0
        # Extract only action token logprobs
        sampling_logprobs_actions = sampling_logprobs[action_mask]
        training_logprobs_actions = training_logprobs[action_mask]

        if len(sampling_logprobs_actions) > 0:
            logprob_diff = sampling_logprobs_actions - training_logprobs_actions
            all_diffs.append(logprob_diff)
            all_sampling_logprobs.append(sampling_logprobs_actions)

    assert all_diffs
    flat_diffs = torch.cat(all_diffs)
    kl_sample_train_v1 = flat_diffs.mean().item()
    kl_sample_train_v2 = 0.5 * (flat_diffs**2).mean().item()

    flat_sampling_logprobs = torch.cat(all_sampling_logprobs)
    entropy_sample = -flat_sampling_logprobs.mean().item()
    return {
        "optim/kl_sample_train_v1": kl_sample_train_v1,
        "optim/kl_sample_train_v2": kl_sample_train_v2,
        "optim/entropy": entropy_sample,
    }


@dataclass(frozen=True)
class KlSampleTrainDetails:
    """Extended sampler-vs-trainer KL diagnostics.

    Attributes:
        metrics (dict[str, float]): Flat metric dict, a superset of
            :func:`compute_kl_sample_train`'s output, with distributional
            statistics over per-token logprob differences and importance
            sampling ratios.
        per_datum (list[dict[str, Any]]): JSON-safe per-datum records with
            per-turn breakdowns, suitable for JSONL export.
    """

    metrics: dict[str, float]
    per_datum: list[dict[str, Any]]


def _mask_runs(action_mask: torch.Tensor) -> list[tuple[int, int]]:
    """Find contiguous runs of True values in a 1D boolean mask.

    Each run corresponds to one assistant action segment in a multi-turn datum.

    Args:
        action_mask (torch.Tensor): 1D boolean tensor over the full datum length.

    Returns:
        list[tuple[int, int]]: ``(start, end)`` index pairs (end exclusive),
            one per contiguous run of True values, in order.
    """
    runs: list[tuple[int, int]] = []
    start: int | None = None
    mask_values: list[bool] = action_mask.tolist()
    for idx, is_action in enumerate(mask_values):
        if is_action and start is None:
            start = idx
        elif not is_action and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask_values)))
    return runs


def compute_kl_sample_train_extended(
    data_D: list[tinker.Datum],
    training_logprobs_D: list[torch.Tensor],
    substep_ids_D: list[int] | None = None,
) -> KlSampleTrainDetails:
    """Compute extended sampler-vs-trainer KL diagnostics.

    Builds on :func:`compute_kl_sample_train` by additionally characterizing
    the *distribution* of per-token logprob differences
    ``diff = sampling_logprob - training_logprob`` (so the importance sampling
    ratio used by the ``importance_sampling`` loss is ``exp(-diff)``), both
    pooled over the batch and per datum / per turn.

    Args:
        data_D (list[tinker.Datum]): List of datums, each containing
            ``logprobs`` and ``mask`` tensors in ``loss_fn_inputs``.
        training_logprobs_D (list[torch.Tensor]): Per-token logprobs from the
            current training model, one tensor per datum.
        substep_ids_D (list[int] | None): Optional optimizer-substep id per
            datum. When given and more than one distinct value is present,
            per-substep mean diffs are emitted as
            ``optim/kl_sample_train_v1/substep_{s}``.

    Returns:
        KlSampleTrainDetails: ``metrics`` (flat floats, superset of
            :func:`compute_kl_sample_train`) and ``per_datum`` (JSON-safe
            records with per-turn segmentation). Datums with zero action
            tokens are skipped in the extended statistics.
    """
    metrics: dict[str, float] = dict(compute_kl_sample_train(data_D, training_logprobs_D))

    diffs_per_datum: list[torch.Tensor] = []
    per_datum: list[dict[str, Any]] = []
    substep_to_diffs: dict[int, list[torch.Tensor]] = defaultdict(list)

    for datum_idx, (datum, training_logprobs) in enumerate(safezip(data_D, training_logprobs_D)):
        sampling_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
        action_mask = datum.loss_fn_inputs["mask"].to_torch() > 0
        full_diff = sampling_logprobs - training_logprobs
        diff = full_diff[action_mask]
        if len(diff) == 0:
            # Datum with no action tokens contributes nothing
            continue

        substep = substep_ids_D[datum_idx] if substep_ids_D is not None else 0
        diffs_per_datum.append(diff)
        substep_to_diffs[substep].append(diff)

        turns: list[dict[str, Any]] = []
        for turn_idx, (start, end) in enumerate(_mask_runs(action_mask)):
            turn_diff = full_diff[start:end]
            turns.append(
                {
                    "turn_idx": turn_idx,
                    "n_tokens": end - start,
                    "mean_diff": turn_diff.mean().item(),
                    "first_token_diff": turn_diff[0].item(),
                }
            )

        is_ratio = torch.exp(-diff)
        per_datum.append(
            {
                "datum_idx": datum_idx,
                "substep": substep,
                "n_action_tokens": int(diff.numel()),
                "mean_diff": diff.mean().item(),
                "p95_diff": torch.quantile(diff, 0.95).item(),
                "max_abs_diff": diff.abs().max().item(),
                "frac_neg": (diff < 0).float().mean().item(),
                "is_ratio_mean": is_ratio.mean().item(),
                "is_ratio_max": is_ratio.max().item(),
                "turns": turns,
            }
        )

    if not diffs_per_datum:
        return KlSampleTrainDetails(metrics=metrics, per_datum=[])

    flat = torch.cat(diffs_per_datum)
    ratio = torch.exp(-flat)
    datum_mean_diffs = torch.tensor([diff.mean().item() for diff in diffs_per_datum])

    metrics.update(
        {
            "optim/logprob_diff/p05": torch.quantile(flat, 0.05).item(),
            "optim/logprob_diff/p50": torch.quantile(flat, 0.50).item(),
            "optim/logprob_diff/p95": torch.quantile(flat, 0.95).item(),
            "optim/logprob_diff/p99": torch.quantile(flat, 0.99).item(),
            "optim/logprob_diff/max_abs": flat.abs().max().item(),
            "optim/logprob_diff/frac_neg": (flat < 0).float().mean().item(),
            "optim/is_ratio/mean": ratio.mean().item(),
            "optim/is_ratio/p99": torch.quantile(ratio, 0.99).item(),
            "optim/is_ratio/max": ratio.max().item(),
            "optim/is_ratio/frac_gt_2": (ratio > 2.0).float().mean().item(),
            "optim/is_ratio/frac_lt_half": (ratio < 0.5).float().mean().item(),
            "optim/kl_datum/p50": torch.quantile(datum_mean_diffs, 0.50).item(),
            "optim/kl_datum/p95": torch.quantile(datum_mean_diffs, 0.95).item(),
            "optim/kl_datum/max": datum_mean_diffs.max().item(),
        }
    )

    if substep_ids_D is not None and len(set(substep_ids_D)) > 1:
        for substep, substep_diffs in sorted(substep_to_diffs.items()):
            metrics[f"optim/kl_sample_train_v1/substep_{substep}"] = (
                torch.cat(substep_diffs).mean().item()
            )

    return KlSampleTrainDetails(metrics=metrics, per_datum=per_datum)


def compute_policy_version_metrics(
    trajectory_groups_P: Sequence[TrajectoryGroup],
    trainer_version_pre: int | None,
) -> dict[str, float]:
    """Aggregate sampler policy-version and retry metadata across rollouts.

    Walks every transition's action (:class:`~tinker_cookbook.completers.TokensWithLogprobs`)
    and collects the optional ``policy_version`` / ``sample_retries`` /
    ``sample_retry_wait_s`` fields. These are populated only by SDK variants
    that expose them (e.g. the Loops tinker shim), so keys are emitted only
    when the corresponding inputs exist.

    Args:
        trajectory_groups_P (Sequence[TrajectoryGroup]): Trajectory groups
            from one training batch.
        trainer_version_pre (int | None): Trainer policy version read before
            the optimizer step, or ``None`` if unavailable.

    Returns:
        dict[str, float]: Possibly-empty dict with keys (each emitted only
            when its inputs exist):
            - ``policy_version/sampler_min`` / ``sampler_max`` / ``frac_missing``
            - ``policy_version/trainer_pre``
            - ``policy_version/lag`` (``trainer_pre - sampler_max``)
            - ``sample_retry/count_total`` / ``wait_total_s`` / ``wait_max_s``
    """
    versions: list[int | None] = []
    retry_counts: list[int] = []
    retry_waits: list[float] = []
    for trajectory_group in trajectory_groups_P:
        for trajectory in trajectory_group.trajectories_G:
            for transition in trajectory.transitions:
                versions.append(transition.ac.policy_version)
                if transition.ac.sample_retries is not None:
                    retry_counts.append(transition.ac.sample_retries)
                if transition.ac.sample_retry_wait_s is not None:
                    retry_waits.append(transition.ac.sample_retry_wait_s)

    metrics: dict[str, float] = {}
    known_versions = [version for version in versions if version is not None]
    if known_versions:
        metrics["policy_version/sampler_min"] = float(min(known_versions))
        metrics["policy_version/sampler_max"] = float(max(known_versions))
        metrics["policy_version/frac_missing"] = 1.0 - len(known_versions) / len(versions)
    if trainer_version_pre is not None:
        metrics["policy_version/trainer_pre"] = float(trainer_version_pre)
        if known_versions:
            metrics["policy_version/lag"] = float(trainer_version_pre - max(known_versions))
    if retry_counts or retry_waits:
        metrics["sample_retry/count_total"] = float(sum(retry_counts))
        metrics["sample_retry/wait_total_s"] = float(sum(retry_waits))
        metrics["sample_retry/wait_max_s"] = max(retry_waits, default=0.0)
    return metrics


def read_trainer_policy_version(training_client: tinker.TrainingClient) -> int | None:
    """Read the trainer's current policy version, if the client exposes one.

    Real Tinker training clients have no ``policy_version`` attribute. On the
    Loops tinker shim it is a property backed by a blocking (~20ms) HTTP GET
    that may raise, so any failure is swallowed and reported as ``None``.

    Args:
        training_client (tinker.TrainingClient): The training client to probe.

    Returns:
        int | None: The policy version, or ``None`` when the attribute is
            missing, ``None``, non-integral, or raises on access.
    """
    try:
        version = getattr(training_client, "policy_version", None)
        if version is None:
            return None
        return int(version)
    except Exception:
        return None


@trace.scope
async def compute_post_kl(
    data_D: list[tinker.Datum], post_sampling_client: tinker.SamplingClient
) -> dict[str, float]:
    """Compute post-update KL divergence metrics.

    Measures how much the policy changed after a training update by computing
    logprobs on the same sequences using a post-update sampling client and
    comparing them against the original sampling logprobs. Reconstructs the
    full token sequence from the shifted inputs and targets before computing
    logprobs.

    Args:
        data_D (list[tinker.Datum]): List of datums containing the original
            sampling logprobs, target tokens, and action masks in
            ``loss_fn_inputs``.
        post_sampling_client (tinker.SamplingClient): A sampling client loaded
            with the post-update model weights.

    Returns:
        dict[str, float]: Dictionary with keys:
            - ``kl_pre_post_v1``: Mean logprob difference (first-order KL estimate).
            - ``kl_pre_post_v2``: Half mean squared logprob difference (second-order KL).
    """
    # Compute logprobs at all data items
    # This is a bit ugly, but we first reconstruct the original sequence from before we did the
    # shifting to get the inputs and targets.
    full_sequence_inputs_D = [
        datum.model_input.append_int(cast(int, datum.loss_fn_inputs["target_tokens"].data[-1]))
        for datum in data_D
    ]
    new_logprobs_D = await asyncio.gather(
        *[
            post_sampling_client.compute_logprobs_async(sequence_input)
            for sequence_input in full_sequence_inputs_D
        ]
    )

    prev_logprobs_D = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in data_D]
    action_masks = [datum.loss_fn_inputs["mask"].to_torch() > 0 for datum in data_D]
    flat_diffs = [
        (prev_logprobs - torch.tensor(new_logprobs[1:]))[action_mask]
        for new_logprobs, prev_logprobs, action_mask in safezip(
            new_logprobs_D, prev_logprobs_D, action_masks
        )
    ]
    flat_diffs = torch.cat(flat_diffs)
    kl_post_v1 = flat_diffs.mean().item()
    kl_post_v2 = 0.5 * (flat_diffs**2).mean().item()

    return {"kl_pre_post_v1": kl_post_v1, "kl_pre_post_v2": kl_post_v2}


@trace.scope
async def incorporate_kl_penalty(
    data_D: list[tinker.Datum],
    base_sampling_client: tinker.SamplingClient,
    kl_penalty_coef: float,
    kl_discount_factor: float,
) -> dict[str, float]:
    """Compute KL penalty against the base model and adjust advantages in-place.

    Computes the per-token logprob difference between the sampling policy and
    the base model, then adjusts each datum's advantages by adding a KL penalty
    term: ``kl_penalty_coef * (avg_kl - per_token_kl)``. When
    ``kl_discount_factor > 0``, the KL penalty is transformed into discounted
    future sums before being added to advantages.

    The KL direction is ``logp_sampled - logp_base``, so ``avg_kl`` represents
    ``-KL[current || base]`` in expectation.

    Args:
        data_D (list[tinker.Datum]): List of datums whose ``advantages`` in
            ``loss_fn_inputs`` will be modified in-place. Must also contain
            ``logprobs``, ``mask``, and ``target_tokens``.
        base_sampling_client (tinker.SamplingClient): A sampling client loaded
            with the base (reference) model weights.
        kl_penalty_coef (float): Coefficient scaling the KL penalty added to
            advantages. Higher values regularize more strongly toward the base
            model.
        kl_discount_factor (float): Discount factor for computing discounted
            future sums of the KL penalty. Set to 0 to disable discounting.

    Returns:
        dict[str, float]: Dictionary with key ``kl_policy_base`` containing the
            mean logprob difference between the sampling policy and base model
            (averaged over action tokens).
    """
    # Compute logprobs at all data items
    full_sequence_inputs_D = [
        datum.model_input.append_int(cast(int, datum.loss_fn_inputs["target_tokens"].data[-1]))
        for datum in data_D
    ]
    base_logprobs_D = await asyncio.gather(
        *[
            base_sampling_client.compute_logprobs_async(sequence_input)
            for sequence_input in full_sequence_inputs_D
        ]
    )
    # compute the logprob differences, zeroed out when the mask == 0
    sampled_logprobs_D = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in data_D]
    float_masks = [datum.loss_fn_inputs["mask"].to_torch().float() for datum in data_D]
    logprob_diffs = [
        (sampled_logprobs - torch.tensor(base_logprobs[1:])) * mask
        for base_logprobs, sampled_logprobs, mask in safezip(
            base_logprobs_D, sampled_logprobs_D, float_masks
        )
    ]
    avg_logp_diff = sum([diff.sum() for diff in logprob_diffs]) / sum(
        [mask.sum() for mask in float_masks]
    )
    for i, datum in enumerate(data_D):
        kl_advantages = kl_penalty_coef * float_masks[i] * (avg_logp_diff - logprob_diffs[i])
        if kl_discount_factor > 0:
            kl_advantages = discounted_future_sum_vectorized(kl_advantages, kl_discount_factor)
        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
            datum.loss_fn_inputs["advantages"].to_torch() + kl_advantages
        )

    return {"kl_policy_base": float(avg_logp_diff)}


def discounted_future_sum_vectorized(x: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute discounted sum of future values for each position.

    For position i, computes: ``sum_{k=0}^{T-1-i} gamma^k * x[i+k]``.
    Uses a single backward pass for O(T) computation.

    Args:
        x (torch.Tensor): 1D tensor of values.
        gamma (float): Discount factor in [0, 1].

    Returns:
        torch.Tensor: 1D tensor of the same shape as ``x``, where each element
            is the discounted sum of current and future values.
    """
    result = torch.empty_like(x)
    running = torch.zeros(1, dtype=x.dtype, device=x.device)
    for t in range(len(x) - 1, -1, -1):
        running = x[t] + gamma * running
        result[t] = running
    return result


def compute_sampling_client_metrics(
    wrapped_trajectory_groups: list[Any],  # WrappedTrajectoryGroup
) -> dict[str, Any]:
    """Compute metrics about sampling clients used to generate trajectory groups.

    Aggregates statistics about which training step each sampling client was at
    when trajectories were generated, and how long sampling took. Useful for
    monitoring staleness of rollout data in asynchronous training.

    Args:
        wrapped_trajectory_groups (list[Any]): List of WrappedTrajectoryGroup
            objects, each having a ``sampling_client_step`` attribute and a
            ``metrics`` dict containing ``time/trajectory_group_worker_loop/total``.

    Returns:
        dict[str, Any]: Dictionary with keys:
            - ``sampling_client/step_max``: Maximum training step across sampling clients.
            - ``sampling_client/step_min``: Minimum training step across sampling clients.
            - ``sampling_client/step_mean``: Mean training step across sampling clients.
            - ``time/sampling_time_max``: Maximum sampling time across groups.
            - ``time/sampling_time_min``: Minimum sampling time across groups.
            - ``time/sampling_time_mean``: Mean sampling time across groups.
    """
    if not wrapped_trajectory_groups:
        return {}
    sampling_client_steps = [
        wrapped_trajectory_group.sampling_client_step
        for wrapped_trajectory_group in wrapped_trajectory_groups
    ]
    sample_times = [
        wrapped_trajectory_group.metrics["time/trajectory_group_worker_loop/total"]
        for wrapped_trajectory_group in wrapped_trajectory_groups
    ]
    metrics = {}
    metrics["sampling_client/step_max"] = max(sampling_client_steps)
    metrics["sampling_client/step_min"] = min(sampling_client_steps)
    metrics["sampling_client/step_mean"] = sum(sampling_client_steps) / len(sampling_client_steps)
    metrics["time/sampling_time_max"] = max(sample_times)
    metrics["time/sampling_time_min"] = min(sample_times)
    metrics["time/sampling_time_mean"] = sum(sample_times) / len(sample_times)
    return metrics

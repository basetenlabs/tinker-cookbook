"""
Tests for the cascading shutdown mechanism in async RL training.

These tests validate that when the dataloader exhausts its data, the shutdown
propagates cleanly through the pipeline without hanging:
  dataloader -> workers -> training loop -> evaluation loop
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tinker_cookbook.rl import train
from tinker_cookbook.rl.train import _AsyncCounter, _Shutdown


@pytest.mark.parametrize("batch_sizes", [(8, 8, 8), (8, 8, 7), (1, 1, 1)])
def test_async_streaming_drains_complete_batches(monkeypatch, tmp_path, batch_sizes):
    """A buffered shutdown marker must not overtake the final complete batch."""

    async def scenario():
        workers_done = asyncio.Event()
        decrement = _AsyncCounter.decrement_and_get

        async def track_worker_exit(counter):
            remaining = await decrement(counter)
            if remaining == 0:
                workers_done.set()
            return remaining

        monkeypatch.setattr(_AsyncCounter, "decrement_and_get", track_worker_exit)
        builders = [Mock() for _ in range(sum(batch_sizes))]
        for builder in builders:
            builder.logging_tags.return_value = ["test"]
        batches = []
        offset = 0
        for size in batch_sizes:
            batches.append(builders[offset : offset + size])
            offset += size
        trained = []

        async def prepare(group_builders, *args, **kwargs):
            # Make later batches and the shutdown marker arrive while the first
            # optimizer batch is in flight, as they do with fast sampling.
            await workers_done.wait()
            trained.extend(group_builders)
            return [], {}

        sampler = Mock()
        client = Mock()
        client.create_sampling_client.return_value = sampler
        client.forward_backward_async = AsyncMock(
            return_value=SimpleNamespace(
                result_async=AsyncMock(return_value=SimpleNamespace(loss_fn_outputs=[]))
            )
        )
        client.optim_step_async = AsyncMock(
            return_value=SimpleNamespace(
                result_async=AsyncMock(return_value=SimpleNamespace(metrics={}))
            )
        )
        monkeypatch.setattr(
            train.checkpoint_utils,
            "save_checkpoint_async",
            AsyncMock(return_value={"sampler_path": "test-sampler"}),
        )
        monkeypatch.setattr(
            train,
            "do_group_rollout_and_filter_constant_reward",
            AsyncMock(side_effect=lambda *args, **kwargs: Mock()),
        )
        monkeypatch.setattr(train, "prepare_minibatch", prepare)
        monkeypatch.setattr(train, "compute_trajectory_metrics", Mock(return_value={}))
        monkeypatch.setattr(
            train,
            "compute_full_batch_metrics_and_get_sampling_client",
            AsyncMock(return_value=(sampler, {})),
        )
        logger = Mock(store=None)
        groups_per_batch = batch_sizes[0]
        config = train.Config(
            learning_rate=1e-5,
            dataset_builder=Mock(),
            model_name="test-model",
            recipe_name="test",
            max_tokens=8,
            log_path=str(tmp_path),
            async_config=train.AsyncConfig(
                groups_per_batch=groups_per_batch,
                max_steps_off_policy=len(batches),
            ),
            stream_minibatch_config=train.StreamMinibatchConfig(
                groups_per_batch=groups_per_batch,
                num_minibatches=1,
            ),
            eval_every=0,
            span_chart_every=0,
            rollout_json_export=False,
        )
        await asyncio.wait_for(
            train.do_async_training(
                start_batch=0,
                end_batch=len(batches),
                num_batches=len(batches),
                config=config,
                training_client=client,
                kl_reference_client=None,
                evaluators=[],
                dataset=SimpleNamespace(get_batch=batches.__getitem__),
                ml_logger=logger,
                tokenizer=Mock(),
            ),
            timeout=5,
        )
        expected_updates = len(builders) // groups_per_batch
        assert client.optim_step_async.await_count == expected_updates
        assert logger.log_metrics.call_count == expected_updates
        assert len(trained) == expected_updates * groups_per_batch
        assert len(set(trained)) == len(trained)
        assert trained == builders[: expected_updates * groups_per_batch]
        if len(builders) % groups_per_batch == 0:
            assert set(trained) == set(builders)

    asyncio.run(scenario())


class TestAsyncCounter:
    def test_decrement_and_get(self):
        async def _test():
            counter = _AsyncCounter(3)
            assert await counter.decrement_and_get() == 2
            assert await counter.decrement_and_get() == 1
            assert await counter.decrement_and_get() == 0

        asyncio.run(_test())

    def test_concurrent_decrements(self):
        """Multiple concurrent decrements should each see a unique value."""

        async def _test():
            counter = _AsyncCounter(100)
            results = await asyncio.gather(*[counter.decrement_and_get() for _ in range(100)])
            # Each decrement should produce a unique value from 0 to 99
            assert sorted(results) == list(range(100))

        asyncio.run(_test())


class TestShutdownCascade:
    def test_dataloader_enqueues_shutdown_sentinels(self):
        """When the dataloader finishes, it should enqueue one _Shutdown per worker."""

        async def _test():
            num_workers = 4
            queue: asyncio.Queue[str | _Shutdown] = asyncio.Queue(maxsize=num_workers)

            for _ in range(num_workers):
                await queue.put(_Shutdown())

            for _ in range(num_workers):
                item = await queue.get()
                assert isinstance(item, _Shutdown)

            assert queue.empty()

        asyncio.run(_test())

    def test_last_worker_signals_training_loop(self):
        """The last worker to exit should enqueue a _Shutdown to the training queue."""

        async def _test():
            num_workers = 3
            counter = _AsyncCounter(num_workers)
            training_queue: asyncio.Queue[str | _Shutdown] = asyncio.Queue()

            for _ in range(num_workers):
                num_alive = await counter.decrement_and_get()
                if num_alive == 0:
                    training_queue.put_nowait(_Shutdown())

            assert training_queue.qsize() == 1
            assert isinstance(await training_queue.get(), _Shutdown)

        asyncio.run(_test())

    def test_full_cascade_no_hang(self):
        """
        Full integration test: wire up all four loops with mock rollouts and verify
        the entire pipeline shuts down cleanly without hanging.
        """

        async def _test():
            num_workers = 2
            num_batches = 2
            items_per_batch = 2

            env_queue: asyncio.Queue[int | _Shutdown] = asyncio.Queue(maxsize=num_workers)
            trajectory_queue: asyncio.Queue[int | _Shutdown | None] = asyncio.Queue()
            dataloader_done = asyncio.Event()
            eval_should_shutdown = asyncio.Event()
            worker_counter = _AsyncCounter(num_workers)
            sampling_updated = asyncio.Event()
            sampling_updated.set()

            loops_completed: list[str] = []

            async def dataloader_loop():
                for batch_idx in range(num_batches):
                    for item_idx in range(items_per_batch):
                        await env_queue.put(batch_idx * items_per_batch + item_idx)
                dataloader_done.set()
                for _ in range(num_workers):
                    await env_queue.put(_Shutdown())
                loops_completed.append("dataloader")

            async def worker_loop():
                while True:
                    item = await env_queue.get()
                    if isinstance(item, _Shutdown):
                        break
                    await asyncio.sleep(0.01)
                    trajectory_queue.put_nowait(item)
                num_alive = await worker_counter.decrement_and_get()
                if num_alive == 0:
                    trajectory_queue.put_nowait(_Shutdown())
                loops_completed.append("worker")

            async def training_loop():
                items_consumed = 0
                target = num_batches * items_per_batch
                while items_consumed < target:
                    item = await trajectory_queue.get()
                    if isinstance(item, _Shutdown):
                        break
                    if item is None:
                        continue
                    items_consumed += 1
                    sampling_updated.set()
                eval_should_shutdown.set()
                sampling_updated.set()
                loops_completed.append("training")

            async def evaluation_loop():
                while not eval_should_shutdown.is_set():
                    await sampling_updated.wait()
                    sampling_updated.clear()
                loops_completed.append("evaluation")

            await asyncio.wait_for(
                asyncio.gather(
                    dataloader_loop(),
                    *[worker_loop() for _ in range(num_workers)],
                    training_loop(),
                    evaluation_loop(),
                ),
                timeout=5.0,
            )

            assert "dataloader" in loops_completed
            assert loops_completed.count("worker") == num_workers
            assert "training" in loops_completed
            assert "evaluation" in loops_completed

        asyncio.run(_test())

    def test_cascade_with_early_shutdown(self):
        """
        When the dataloader has fewer items than the training loop expects,
        the _Shutdown sentinel should still propagate and prevent hanging.
        """

        async def _test():
            num_workers = 2
            num_dataloader_batches = 1
            items_per_batch = 2
            training_loop_target = 10  # Expects more than dataloader provides

            env_queue: asyncio.Queue[int | _Shutdown] = asyncio.Queue(maxsize=num_workers)
            trajectory_queue: asyncio.Queue[int | _Shutdown | None] = asyncio.Queue()
            eval_should_shutdown = asyncio.Event()
            worker_counter = _AsyncCounter(num_workers)
            sampling_updated = asyncio.Event()
            sampling_updated.set()

            async def dataloader_loop():
                for batch_idx in range(num_dataloader_batches):
                    for item_idx in range(items_per_batch):
                        await env_queue.put(batch_idx * items_per_batch + item_idx)
                for _ in range(num_workers):
                    await env_queue.put(_Shutdown())

            async def worker_loop():
                while True:
                    item = await env_queue.get()
                    if isinstance(item, _Shutdown):
                        break
                    trajectory_queue.put_nowait(item)
                num_alive = await worker_counter.decrement_and_get()
                if num_alive == 0:
                    trajectory_queue.put_nowait(_Shutdown())

            async def training_loop():
                i_batch = 0
                while i_batch < training_loop_target:
                    item = await trajectory_queue.get()
                    if isinstance(item, _Shutdown):
                        break
                    if item is None:
                        continue
                    i_batch += 1
                eval_should_shutdown.set()
                sampling_updated.set()

            async def evaluation_loop():
                while not eval_should_shutdown.is_set():
                    await sampling_updated.wait()
                    sampling_updated.clear()

            # Should not hang — shutdown cascade terminates all loops
            await asyncio.wait_for(
                asyncio.gather(
                    dataloader_loop(),
                    *[worker_loop() for _ in range(num_workers)],
                    training_loop(),
                    evaluation_loop(),
                ),
                timeout=5.0,
            )

        asyncio.run(_test())

    def test_requeue_skipped_during_shutdown(self):
        """
        When the dataloader is done, stale samples should be discarded
        rather than requeued (to avoid deadlocking on a full bounded queue).
        """
        dataloader_done = asyncio.Event()

        requeue_attempted = False
        discard_count = 0

        def filter_stale(is_stale: bool) -> bool:
            nonlocal requeue_attempted, discard_count
            if is_stale:
                if dataloader_done.is_set():
                    discard_count += 1
                else:
                    requeue_attempted = True
                return False
            return True

        # Before dataloader is done: stale items should attempt requeue
        filter_stale(is_stale=True)
        assert requeue_attempted

        # After dataloader is done: stale items should be discarded
        requeue_attempted = False
        dataloader_done.set()
        filter_stale(is_stale=True)
        assert not requeue_attempted
        assert discard_count == 1

    def test_none_items_pass_through_during_shutdown(self):
        """
        None items (failed rollouts) should be skipped, and _Shutdown should
        still be received even if preceded by None items.
        """

        async def _test():
            queue: asyncio.Queue[int | _Shutdown | None] = asyncio.Queue()

            queue.put_nowait(None)
            queue.put_nowait(None)
            queue.put_nowait(42)
            queue.put_nowait(None)
            queue.put_nowait(_Shutdown())

            received_items = []
            while True:
                item = await queue.get()
                if isinstance(item, _Shutdown):
                    break
                if item is None:
                    continue
                received_items.append(item)

            assert received_items == [42]

        asyncio.run(_test())

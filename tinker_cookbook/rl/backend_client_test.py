"""Caller-supplied clients must not create a second native service session."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tinker_cookbook.rl import train


class ReachedTrainingClient(Exception):
    pass


@pytest.mark.parametrize("injected", [False, True])
def test_main_uses_supplied_client_or_native_default(monkeypatch, tmp_path, injected):
    create = AsyncMock(side_effect=ReachedTrainingClient)
    service = SimpleNamespace(create_lora_training_client_async=create)
    factory = Mock(return_value=service)
    monkeypatch.setattr(train.tinker, "ServiceClient", factory)
    monkeypatch.setattr(
        train.ml_log,
        "setup_logging",
        lambda **kwargs: SimpleNamespace(store=None, get_logger_url=lambda: None),
    )
    monkeypatch.setattr(train.model_info, "warn_if_renderer_not_recommended", lambda *args: None)
    config = train.Config(
        dataset_builder=None,
        model_name="fixture",
        recipe_name="test",
        learning_rate=1e-5,
        max_tokens=None,
        log_path=str(tmp_path),
    )
    with pytest.raises(ReachedTrainingClient):
        asyncio.run(train.main(config, service_client=service if injected else None))
    create.assert_awaited_once()
    assert factory.call_count == (0 if injected else 1)

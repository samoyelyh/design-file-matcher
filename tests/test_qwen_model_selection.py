import json

import pytest

from app.main import _configure_batch_engine
from app.models import Batch
from app.services.qwen_production import _selected_model


MODELS = ["qwen3-vl:4b-instruct", "qwen3-vl:8b-instruct"]


@pytest.mark.parametrize("engine", ["PHASH", "DINO", "LIGHTGLUE_COLOR", "QWEN"])
def test_formal_batch_rejects_retired_matchers(engine):
    batch = Batch(root_path="C:/images", product_category="test", status="SCANNED")
    with pytest.raises(ValueError, match="VisionMatch AI"):
        _configure_batch_engine(batch, engine)


def test_worker_uses_batch_selected_model_not_global_preference():
    batch = Batch(
        root_path="C:/images",
        product_category="test",
        status="SCANNED",
        match_engine="QWEN",
        matching_model=MODELS[0],
        match_config_snapshot=json.dumps({"model": MODELS[0]}),
    )
    state = {"reachable": True, "selected_model": MODELS[1], "vision_models": MODELS}

    assert _selected_model(batch, state) == MODELS[0]


def test_worker_rejects_a_selected_model_that_is_no_longer_installed():
    batch = Batch(
        root_path="C:/images",
        product_category="test",
        status="SCANNED",
        matching_model=MODELS[0],
        match_config_snapshot=json.dumps({"model": MODELS[0]}),
    )
    state = {"reachable": True, "selected_model": MODELS[1], "vision_models": [MODELS[1]]}

    with pytest.raises(RuntimeError, match="未就绪或不支持图片输入"):
        _selected_model(batch, state)

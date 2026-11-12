import pytest

from app.services.qwen_fusion import (
    fused_score,
    lightglue_color_score,
    qwen_support_score,
)


def test_qwen_rejection_confidence_is_inverted_to_match_support():
    assert qwen_support_score(True, .82) == .82
    assert qwen_support_score(False, .82) == pytest.approx(.18)


def test_fusion_uses_qwen_and_lightglue_color_weights():
    assert fused_score(.8, .6, qwen_weight=.7, lightglue_color_weight=.3) == pytest.approx(.74)


def test_fusion_renormalizes_when_auxiliary_color_score_is_unavailable():
    assert fused_score(.63, None, qwen_weight=.7, lightglue_color_weight=.3) == pytest.approx(.63)


def test_lightglue_color_score_matches_existing_engine_weights():
    assert lightglue_color_score(.8, .6) == pytest.approx(.3 * .8 + .7 * .6)
    assert lightglue_color_score(.8, None) == pytest.approx(.3 * .8)
    assert lightglue_color_score(None, .6) is None

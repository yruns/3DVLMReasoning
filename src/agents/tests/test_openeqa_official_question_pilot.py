from __future__ import annotations

import json
from pathlib import Path

from agents.examples.openeqa_official_question_pilot import (
    apply_force_selection,
    derive_eval_session_id,
)


def test_force_selection_filters_samples_to_expected_set(tmp_path: Path) -> None:
    samples = [
        {"question_id": "q3", "clip_id": "scene-a"},
        {"question_id": "q1", "clip_id": "scene-a"},
        {"question_id": "q2", "clip_id": "scene-b"},
    ]
    force_selection_path = tmp_path / "official_selected_questions.json"
    force_selection_path.write_text(
        json.dumps(
            [
                {"question_id": "q1"},
                {"question_id": "q2"},
            ]
        ),
        encoding="utf-8",
    )

    filtered = apply_force_selection(
        samples,
        force_selection_path=force_selection_path,
        live_question_ids={"q1", "q2", "q3"},
    )

    assert [sample["question_id"] for sample in filtered] == ["q1", "q2"]


def test_derive_eval_session_id_is_stable_and_separates_prompt_variants() -> None:
    output_root = Path("/tmp/v15_eval")

    default_a = derive_eval_session_id(
        output_root=output_root,
        enable_temporal_fan=False,
    )
    default_b = derive_eval_session_id(
        output_root=output_root,
        enable_temporal_fan=False,
    )
    fan = derive_eval_session_id(
        output_root=output_root,
        enable_temporal_fan=True,
    )

    assert default_a == default_b
    assert default_a.startswith("v15_")
    assert len(default_a) == len("v15_" + "0" * 16)
    assert fan != default_a

"""VG pack tools: per-tool tests + FAIL-LOUD gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import KeyframeEvidence, Stage2EvidenceBundle
from agents.packs.vg_embodiedscan.ctx import (
    Proposal,
    ProposalFrameView,
    VgEmbodiedScanCtx,
)
from agents.packs.vg_embodiedscan.tools import build_vg_tools
from agents.runtime.base import Stage2RuntimeState


def _runtime(tmp_path: Path) -> Stage2RuntimeState:
    annotated = tmp_path / "ann"
    annotated.mkdir()
    bundle = Stage2EvidenceBundle(
        keyframes=[
            KeyframeEvidence(keyframe_idx=0, image_path="a.png", frame_id=10),
            KeyframeEvidence(keyframe_idx=1, image_path="b.png", frame_id=11),
        ]
    )
    rs = Stage2RuntimeState(bundle=bundle)
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.task_ctx = VgEmbodiedScanCtx(
        proposal_pool_source="vdetr",
        proposals=[
            Proposal(id=0, bbox_3d_9dof=[0] * 9, category="chair", score=0.9),
            Proposal(id=1, bbox_3d_9dof=[1] * 9, category="desk", score=0.8),
            Proposal(id=2, bbox_3d_9dof=[2] * 9, category="chair", score=0.7),
        ],
        frame_index={10: [0, 1], 11: [1, 2]},
        proposal_index={0: [10], 1: [10, 11], 2: [11]},
        annotated_image_dir=annotated,
    )
    return rs


def test_list_keyframes_with_proposals_gates_on_skill(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    tool = next(
        t for t in build_vg_tools(rs) if t.name == "list_keyframes_with_proposals"
    )
    response = tool.invoke({})
    assert response.startswith("ERROR")
    assert "vg-grounding-playbook" in response


def test_list_keyframes_with_proposals_returns_structured(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(
        t for t in build_vg_tools(rs) if t.name == "list_keyframes_with_proposals"
    )
    payload = json.loads(tool.invoke({}))
    assert len(payload) == 2
    assert payload[0]["frame_id"] == 10
    assert payload[0]["visible_proposal_ids"] == [0, 1]
    assert payload[0]["annotated_image"].endswith("/ann/frame_10.png")


def test_view_keyframe_marked_returns_image_content(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    # create a fake marked image
    marked = rs.task_ctx.annotated_image_dir / "frame_10.png"
    marked.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header

    tool = next(t for t in build_vg_tools(rs) if t.name == "view_keyframe_marked")
    response = tool.invoke({"frame_id": 10})
    assert "frame_10.png" in response
    assert "visible_proposals" in response
    assert "[0, 1]" in response or "0, 1" in response


def test_view_keyframe_marked_unknown_frame_errors(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "view_keyframe_marked")
    response = tool.invoke({"frame_id": 999})
    assert response.startswith("ERROR")


def test_view_keyframe_marked_image_drained_into_evidence_update(
    tmp_path: Path,
) -> None:
    """End-to-end: vg_pending_images queued by view_keyframe_marked must be
    drained into the chassis's next-turn user message via
    build_evidence_update_message."""

    from agents.core.agent_config import Stage2DeepAgentConfig
    from agents.runtime.deepagents_agent import DeepAgentsStage2Runtime

    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    # Pre-mark all bundle keyframe images as already-seen so the only
    # NEW image picked up is the marked one queued by view_keyframe_marked.
    for kf in rs.bundle.keyframes:
        rs.seen_image_paths.add(kf.image_path)

    # write a real PNG so chassis image_to_data_url can decode it
    from PIL import Image

    marked = rs.task_ctx.annotated_image_dir / "frame_10.png"
    Image.new("RGB", (4, 4), color=(0, 0, 0)).save(marked, format="PNG")

    tool = next(t for t in build_vg_tools(rs) if t.name == "view_keyframe_marked")
    tool.invoke({"frame_id": 10})

    # The queue must have been populated.
    assert str(marked) in rs.bundle.extra_metadata["vg_pending_images"]

    # Now drain via the chassis injector.
    runtime = DeepAgentsStage2Runtime(config=Stage2DeepAgentConfig())
    msg = runtime.build_evidence_update_message(rs)
    assert msg is not None, "expected an evidence-update message but got None"
    # The path should be present in either the rendered text or the
    # multimodal image_url block; check both representations.
    serialized = str(msg.content)
    assert str(marked) in serialized or any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for part in (msg.content if isinstance(msg.content, list) else [])
    )
    # And the queue must be drained so we don't re-inject next turn.
    assert rs.bundle.extra_metadata["vg_pending_images"] == []


def test_inspect_proposal_returns_metadata_and_frames(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "inspect_proposal")
    payload = json.loads(tool.invoke({"proposal_id": 1}))
    assert payload["proposal_id"] == 1
    assert payload["category"] == "desk"
    assert payload["score"] == 0.8
    assert payload["frames_appeared"] == [10, 11]
    assert payload["bbox_3d_9dof"] == [1] * 9


def test_inspect_proposal_unknown_id_errors(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "inspect_proposal")
    response = tool.invoke({"proposal_id": 99})
    assert response.startswith("ERROR")


def test_find_proposals_by_category_lists_ids(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "find_proposals_by_category")
    payload = json.loads(tool.invoke({"category": "chair"}))
    assert payload["proposal_ids"] == [0, 2]


def test_find_proposals_by_category_unknown_returns_empty(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "find_proposals_by_category")
    payload = json.loads(tool.invoke({"category": "spaceship"}))
    assert payload["proposal_ids"] == []
    assert "available_categories" in payload


def test_compare_proposals_spatial_closest_to(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    # set proposal centers far apart so order is deterministic
    rs.task_ctx.proposals = [
        Proposal(
            id=0, bbox_3d_9dof=[0, 0, 0, 1, 1, 1, 0, 0, 0], category="chair", score=0.9
        ),
        Proposal(
            id=1, bbox_3d_9dof=[5, 5, 5, 1, 1, 1, 0, 0, 0], category="chair", score=0.7
        ),
        Proposal(
            id=2,
            bbox_3d_9dof=[10, 10, 10, 1, 1, 1, 0, 0, 0],
            category="desk",
            score=0.8,
        ),
    ]
    tool = next(t for t in build_vg_tools(rs) if t.name == "compare_proposals_spatial")
    payload = json.loads(
        tool.invoke(
            {
                "candidate_ids": [0, 1],
                "anchor_id": 2,
                "relation": "closest_to",
            }
        )
    )
    assert payload["ranked_ids"] == [1, 0]


def test_compare_proposals_spatial_farthest_from(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    rs.task_ctx.proposals = [
        Proposal(
            id=0, bbox_3d_9dof=[0, 0, 0, 1, 1, 1, 0, 0, 0], category="chair", score=0.9
        ),
        Proposal(
            id=1, bbox_3d_9dof=[5, 5, 5, 1, 1, 1, 0, 0, 0], category="chair", score=0.7
        ),
        Proposal(
            id=2,
            bbox_3d_9dof=[10, 10, 10, 1, 1, 1, 0, 0, 0],
            category="desk",
            score=0.8,
        ),
    ]
    tool = next(t for t in build_vg_tools(rs) if t.name == "compare_proposals_spatial")
    payload = json.loads(
        tool.invoke(
            {
                "candidate_ids": [0, 1],
                "anchor_id": 2,
                "relation": "farthest_from",
            }
        )
    )
    assert payload["ranked_ids"] == [0, 1]


def test_compare_proposals_spatial_above_uses_z_axis(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    rs.task_ctx.proposals = [
        Proposal(
            id=0,
            bbox_3d_9dof=[0, 0, 3, 1, 1, 1, 0, 0, 0],
            category="cabinet",
            score=0.9,
        ),
        Proposal(
            id=1,
            bbox_3d_9dof=[0, 0, 1, 1, 1, 1, 0, 0, 0],
            category="cabinet",
            score=0.7,
        ),
        Proposal(
            id=2,
            bbox_3d_9dof=[0, 0, 0, 1, 1, 1, 0, 0, 0],
            category="refrigerator",
            score=0.8,
        ),
    ]
    tool = next(t for t in build_vg_tools(rs) if t.name == "compare_proposals_spatial")
    payload = json.loads(
        tool.invoke(
            {
                "candidate_ids": [0, 1],
                "anchor_id": 2,
                "relation": "above",
            }
        )
    )
    assert payload["ranked_ids"] == [0, 1]
    assert payload["vertical_offsets"] == [3.0, 1.0]


def test_compare_proposals_spatial_below_uses_z_axis(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    rs.task_ctx.proposals = [
        Proposal(
            id=0, bbox_3d_9dof=[0, 0, -2, 1, 1, 1, 0, 0, 0], category="box", score=0.9
        ),
        Proposal(
            id=1, bbox_3d_9dof=[0, 0, 2, 1, 1, 1, 0, 0, 0], category="box", score=0.7
        ),
        Proposal(
            id=2, bbox_3d_9dof=[0, 0, 0, 1, 1, 1, 0, 0, 0], category="table", score=0.8
        ),
    ]
    tool = next(t for t in build_vg_tools(rs) if t.name == "compare_proposals_spatial")
    payload = json.loads(
        tool.invoke(
            {
                "candidate_ids": [0, 1],
                "anchor_id": 2,
                "relation": "below",
            }
        )
    )
    assert payload["ranked_ids"] == [0, 1]
    assert payload["vertical_offsets"] == [-2.0, 2.0]


def test_compare_proposals_spatial_unknown_relation_errors(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "compare_proposals_spatial")
    response = tool.invoke(
        {
            "candidate_ids": [0, 1],
            "anchor_id": 2,
            "relation": "left_of",
        }
    )
    assert response.startswith("ERROR")


# ---------------------------------------------------------------------------
# CVRA (find_proposals_by_category with clip_visible augmentation)
#
# These tests exercise the M2 canonical contract: feature flag,
# label_hits + clip_visible_aug shapes, dynamic K_AUG cap, dedup,
# rank fallback, cvra_exhausted flag, and the integration synthetic
# that mimics a label-mismatch case end-to-end.
# ---------------------------------------------------------------------------


@dataclass
class _StubClipScore:
    """Mirror of agents.packs.vg_embodiedscan.clip_provider.ClipScore."""

    proposal_id: int
    frame_id: int
    clip_score: float
    raw_rgb_path: Path
    bbox_2d: tuple[int, int, int, int]


class _StubClipProvider:
    """Test stub returning hard-coded scores keyed by (category_lower, pid)."""

    def __init__(self, scores: dict[tuple[str, int], float]):
        self._scores = scores
        self.calls: list[tuple[str, list[int]]] = []

    def score(self, category, requests):  # noqa: ANN001 — duck type
        self.calls.append(
            (category, [int(r.proposal_id) for r in requests])
        )
        out = []
        for r in requests:
            s = float(self._scores.get((category.strip().lower(), int(r.proposal_id)), 0.0))
            out.append(
                _StubClipScore(
                    proposal_id=int(r.proposal_id),
                    frame_id=int(r.frame_id),
                    clip_score=s,
                    raw_rgb_path=r.raw_rgb_path,
                    bbox_2d=r.bbox_2d,
                )
            )
        return out


def _cvra_runtime(
    tmp_path: Path,
    *,
    use_clip_visible_aug: bool = True,
    tau: float = 0.18,
    k_aug: int = 5,
    scores: dict[tuple[str, int], float] | None = None,
    parser_categories_by_rank: dict[int, list[str]] | None = None,
) -> Stage2RuntimeState:
    """Build a runtime + ctx tuned for CVRA tests.

    Pool layout:
      proposal 0: 'monitor'  (visible in frame 10)  — GT-overlap mislabeled
      proposal 1: 'desk'     (visible in frame 10, 11)  — true-label "desk"
      proposal 2: 'lamp'     (visible in frame 11) — distractor
      proposal 3: 'doorframe'(visible in frame 12) — out-of-seen-set GT-mislabel
    """
    annotated = tmp_path / "ann"
    annotated.mkdir(exist_ok=True)
    extra_metadata: dict = {}
    if parser_categories_by_rank is not None:
        extra_metadata["parser_categories_by_rank"] = {
            str(rank): cats for rank, cats in parser_categories_by_rank.items()
        }
    bundle = Stage2EvidenceBundle(
        scene_id="scene_test",
        keyframes=[
            KeyframeEvidence(keyframe_idx=0, image_path="a.png", frame_id=10),
            KeyframeEvidence(keyframe_idx=1, image_path="b.png", frame_id=11),
        ],
        extra_metadata=extra_metadata,
    )
    rs = Stage2RuntimeState(bundle=bundle)
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.skills_loaded.add("vg-grounding-playbook")
    rs.use_clip_visible_aug = use_clip_visible_aug
    rs.clip_visible_tau = tau
    rs.clip_visible_k_aug = k_aug
    rs.clip_visible_provider = _StubClipProvider(scores or {})

    fv = lambda pid, fid: ProposalFrameView(  # noqa: E731
        proposal_id=pid,
        frame_id=fid,
        bbox_2d=(10 * fid, 10 * fid, 10 * fid + 50, 10 * fid + 30),
        raw_rgb_path=Path(f"data/nr3d/scannet/scene_test/raw/{fid:06d}-rgb.png"),
        visibility_weight=1.0 - 0.05 * fid,
    )

    rs.task_ctx = VgEmbodiedScanCtx(
        proposal_pool_source="conceptgraph",
        proposals=[
            Proposal(
                id=0, bbox_3d_9dof=[0] * 9, category="monitor", score=0.9,
                frame_views={10: fv(0, 10)},
            ),
            Proposal(
                id=1, bbox_3d_9dof=[1] * 9, category="desk", score=0.8,
                frame_views={10: fv(1, 10), 11: fv(1, 11)},
            ),
            Proposal(
                id=2, bbox_3d_9dof=[2] * 9, category="lamp", score=0.7,
                frame_views={11: fv(2, 11)},
            ),
            Proposal(
                id=3, bbox_3d_9dof=[3] * 9, category="doorframe", score=0.6,
                frame_views={12: fv(3, 12)},  # frame 12 NOT in seen-set
            ),
        ],
        frame_index={10: [0, 1], 11: [1, 2], 12: [3]},
        proposal_index={0: [10], 1: [10, 11], 2: [11], 3: [12]},
        annotated_image_dir=annotated,
    )
    return rs


def _invoke_find(rs: Stage2RuntimeState, category: str) -> dict:
    tool = next(
        t for t in build_vg_tools(rs) if t.name == "find_proposals_by_category"
    )
    response = tool.invoke({"category": category})
    return json.loads(response)


def test_cvra_disabled_default(tmp_path: Path) -> None:
    """Flag off -> response shape is the legacy {category, proposal_ids,
    available_categories} only; no CVRA fields."""
    rs = _cvra_runtime(tmp_path, use_clip_visible_aug=False)
    payload = _invoke_find(rs, "desk")
    assert payload["proposal_ids"] == [1]
    assert "clip_visible_aug" not in payload
    assert "label_hits" not in payload
    assert "rank_fallback_used" not in payload
    assert "cvra_exhausted" not in payload


def test_cvra_label_hits_unchanged(tmp_path: Path) -> None:
    """Flag on, label match exists -> label_hits primary; clip aug may
    add at most K=3 extras (label-hit case dynamic cap), and same id is
    not double-counted."""
    rs = _cvra_runtime(
        tmp_path,
        scores={
            ("desk", 0): 0.5,  # monitor-labelled but high score for desk
            ("desk", 2): 0.4,  # lamp false positive
        },
    )
    payload = _invoke_find(rs, "desk")
    assert payload["label_hits"] == [{"proposal_id": 1, "source": "label_exact"}]
    assert 1 not in [item["proposal_id"] for item in payload["clip_visible_aug"]]
    aug_ids = [item["proposal_id"] for item in payload["clip_visible_aug"]]
    assert 0 in aug_ids and 2 in aug_ids
    assert payload["proposal_ids"][0] == 1  # label hit listed first
    assert payload["rank_fallback_used"] is False
    assert payload["cvra_exhausted"] is False


def test_cvra_visible_set_only(tmp_path: Path) -> None:
    """Augmented ids are restricted to proposals visible in the agent's
    cumulative seen frames (not the full scene)."""
    rs = _cvra_runtime(
        tmp_path,
        scores={
            ("desk", 0): 0.5,
            ("desk", 3): 0.95,  # high score on a NOT-seen proposal (frame 12)
        },
    )
    payload = _invoke_find(rs, "desk")
    aug_ids = {item["proposal_id"] for item in payload["clip_visible_aug"]}
    assert 3 not in aug_ids  # never returned even with score 0.95
    assert 0 in aug_ids
    assert payload["n_visible_set"] == 3  # proposals 0, 1, 2 in seen frames


def test_cvra_tau_filters_below_threshold(tmp_path: Path) -> None:
    """Only proposals with clip_score >= TAU are augmented."""
    rs = _cvra_runtime(
        tmp_path,
        tau=0.30,
        scores={
            ("chair", 0): 0.25,  # below tau
            ("chair", 2): 0.40,  # above tau
        },
    )
    payload = _invoke_find(rs, "chair")
    aug_ids = [item["proposal_id"] for item in payload["clip_visible_aug"]]
    assert aug_ids == [2]  # only above-TAU proposal


def test_cvra_k_aug_dynamic_fill(tmp_path: Path) -> None:
    """When label_hits exists, dynamic cap = 3. Otherwise cap = 5
    (or k_aug, whichever is smaller)."""
    # label_hits-present case: 'desk' exists in pool, proposals 0, 2 are
    # CLIP-visible; cap should be 3 — both fit, but no overflow tested
    rs = _cvra_runtime(
        tmp_path,
        scores={
            ("desk", 0): 0.40,
            ("desk", 2): 0.30,
        },
    )
    payload = _invoke_find(rs, "desk")
    assert len(payload["clip_visible_aug"]) <= 3
    assert payload["label_hits"][0]["proposal_id"] == 1

    # no-label-hits case: 'fridge' not in pool; CLIP scores high on 0,1,2
    # (3 visible) — cap 5 lets all 3 through.
    rs2 = _cvra_runtime(
        tmp_path,
        scores={
            ("fridge", 0): 0.5,
            ("fridge", 1): 0.4,
            ("fridge", 2): 0.3,
        },
    )
    payload2 = _invoke_find(rs2, "fridge")
    assert payload2["label_hits"] == []
    aug_ids = sorted(item["proposal_id"] for item in payload2["clip_visible_aug"])
    assert aug_ids == [0, 1, 2]


def test_cvra_dedup_label_and_aug(tmp_path: Path) -> None:
    """A proposal that label-matches must NOT also appear in
    clip_visible_aug (dedup). Even if its score is the highest."""
    rs = _cvra_runtime(
        tmp_path,
        scores={
            ("desk", 1): 0.99,  # label hit AND highest score
            ("desk", 0): 0.5,
            ("desk", 2): 0.3,
        },
    )
    payload = _invoke_find(rs, "desk")
    aug_pids = [item["proposal_id"] for item in payload["clip_visible_aug"]]
    assert 1 not in aug_pids
    assert 1 in [h["proposal_id"] for h in payload["label_hits"]]


def test_cvra_rank_fallback_fires(tmp_path: Path) -> None:
    """If rank-1 returns empty (no label, no clip), rank-2 is tried."""
    rs = _cvra_runtime(
        tmp_path,
        parser_categories_by_rank={1: ["fridge"], 2: ["lamp"]},
        scores={
            # rank-1 'fridge' has no signal
            ("fridge", 0): 0.0, ("fridge", 1): 0.0, ("fridge", 2): 0.0,
            # rank-2 'lamp' label-matches proposal 2
        },
    )
    payload = _invoke_find(rs, "fridge")
    assert payload["rank_fallback_used"] is True
    assert payload["label_hits"] == [{"proposal_id": 2, "source": "label_exact"}]
    assert payload["cvra_exhausted"] is False


def test_cvra_exhausted_flag(tmp_path: Path) -> None:
    """All ranks dry -> cvra_exhausted=True, proposal_ids=[]."""
    rs = _cvra_runtime(
        tmp_path,
        parser_categories_by_rank={1: ["fridge"], 2: ["oven"], 3: ["microwave"]},
        # no scores keyed to any of these categories
    )
    payload = _invoke_find(rs, "fridge")
    assert payload["cvra_exhausted"] is True
    assert payload["proposal_ids"] == []
    assert payload["label_hits"] == []
    assert payload["clip_visible_aug"] == []


def test_cvra_metadata_schema_full(tmp_path: Path) -> None:
    """The augmented entry shape must match the canonical §H spec:
    proposal_id, clip_score, rank_used, via_category, source_frame_id,
    source='clip_visible', cvra_category_source='rank<n>'."""
    rs = _cvra_runtime(
        tmp_path,
        scores={("monitor", 0): 0.5},
    )
    payload = _invoke_find(rs, "monitor")
    assert payload["label_hits"][0] == {"proposal_id": 0, "source": "label_exact"}
    # Aug should be empty here because proposal 0 is the label hit and dedup excludes it
    aug = payload["clip_visible_aug"]
    assert aug == []  # dedup case
    # Now flip: query 'desk' with monitor proposal augmented
    rs2 = _cvra_runtime(
        tmp_path,
        scores={("desk", 0): 0.5},
    )
    payload2 = _invoke_find(rs2, "desk")
    assert len(payload2["clip_visible_aug"]) == 1
    item = payload2["clip_visible_aug"][0]
    expected_keys = {
        "proposal_id", "clip_score", "rank_used", "via_category",
        "source_frame_id", "source", "cvra_category_source",
    }
    assert expected_keys.issubset(item.keys())
    assert item["proposal_id"] == 0
    assert item["source"] == "clip_visible"
    assert item["rank_used"] == 1
    assert item["via_category"] == "desk"
    assert item["cvra_category_source"] == "rank1"


def test_cvra_integration_synthetic_label_mismatched_gt_overlap(
    tmp_path: Path,
) -> None:
    """Integration synthetic per canonical §J item 9.

    Setup mirrors a real `mask3d_label_mismatch` case:
      - GT category is 'desk', GT-overlap proposal exists in the pool
        but Mask3D mislabeled it as 'monitor' (proposal 0)
      - Agent has frames 10 + 11 in the cumulative seen set
      - CLIP scores 'desk' high against the GT-overlap crop (0.45)
      - Pool also has a true 'desk' (proposal 1) and irrelevant 'lamp'

    Assertion: the augmented list contains the GT-overlap proposal_id=0
    so that the M1c retrieval-recall PASS criterion
    (clip_visible_aug_included_gt_overlap=True) can be verified.
    """
    rs = _cvra_runtime(
        tmp_path,
        tau=0.18,
        k_aug=5,
        scores={
            ("desk", 0): 0.45,  # GT-overlap, mislabeled `monitor`
            ("desk", 2): 0.10,  # below-TAU lamp
        },
    )
    payload = _invoke_find(rs, "desk")
    label_pids = [h["proposal_id"] for h in payload["label_hits"]]
    aug_pids = [item["proposal_id"] for item in payload["clip_visible_aug"]]
    assert 1 in label_pids                   # true desk picked up by label
    assert 0 in aug_pids                      # GT-overlap proposal augmented
    assert 2 not in aug_pids                  # lamp dropped by TAU
    # The union (label + aug) is what the agent sees as candidates
    assert payload["proposal_ids"][0] == 1
    assert 0 in payload["proposal_ids"]
    # The retrieval-recall flag for the funnel to read
    assert any(
        item["proposal_id"] == 0
        and item["clip_score"] >= 0.18
        for item in payload["clip_visible_aug"]
    )

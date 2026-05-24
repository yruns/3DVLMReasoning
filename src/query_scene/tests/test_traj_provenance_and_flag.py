"""Phase-3 / must-fix-review tests: camera trajectory provenance + flag.

Covers:

- ``_find_camera_trajectory_file`` and ``_enumerate_camera_trajectory_files``
  probe both ``scene_path`` and ``scene_path.parent`` for pack dirs, so the
  NR3D layout (``.../scene_XXXX/conceptgraph`` as scene_path with packs at
  ``.../scene_XXXX/pack_*``) is supported.
- ``_read_camera_trajectory_json`` explicitly recognizes and rejects the
  position-only dict format ``{"0": [x,y,z], ...}`` rather than silently
  returning zero 4x4 poses and falsely claiming the trajectory loaded.
- ``_load_camera_poses`` falls through from a rejected pack JSON to the
  raw/traj.txt source when both exist.
- ``_force_legacy_world_hard`` collapses every constraint back to WORLD /
  HARD so ``viewpoint_aware=False`` causes strict no-drift.
- ``_guard_viewpoint_traj_available`` RAISES (does not warn) when
  viewpoint_aware is on, a viewer-frame constraint is present, and no
  4x4 camera poses are loaded.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from query_scene.core import (
    ExecutionPolicy,
    GroundingQuery,
    HypothesisOutputV1,
    QueryHypothesis,
    QueryNode,
    ReferenceFrame,
    SpatialConstraint,
    ViewpointContext,
    ViewpointKind,
)
from query_scene.keyframe_selector import KeyframeSelector


class _SelectorStub(KeyframeSelector):
    """Just enough state for the methods under test."""

    def __init__(self, scene_path: Path, camera_poses: list | None = None) -> None:
        self.scene_path = scene_path
        self.stride = 1
        self.camera_poses = camera_poses or []
        self.camera_trajectory_path = None
        self.scene_categories = ["cabinet", "door"]


def _identity_pose() -> np.ndarray:
    return np.eye(4)


def _identity_pose_list() -> list[list[float]]:
    return _identity_pose().tolist()


def _write_traj_txt_16(path: Path, n: int = 3) -> None:
    line = " ".join(
        [
            "1", "0", "0", "0",
            "0", "1", "0", "0",
            "0", "0", "1", "0",
            "0", "0", "0", "1",
        ]
    )
    path.write_text(("\n".join([line] * n)) + "\n")


def _wrap(gq: GroundingQuery) -> HypothesisOutputV1:
    return HypothesisOutputV1(
        parse_mode="single",
        hypotheses=[
            QueryHypothesis(
                kind="direct", rank=1, grounding_query=gq, lexical_hints=[]
            )
        ],
    )


def _viewer_grounding() -> GroundingQuery:
    ctx = ViewpointContext(
        id="vp1",
        kind=ViewpointKind.FACING_ANCHOR,
        facing_anchor=QueryNode(categories=["door"], node_id="vp_door_facing"),
        raw_phrase="facing the door",
    )
    sc = SpatialConstraint(
        relation="right_of",
        anchors=[QueryNode(categories=["door"], node_id="anchor")],
        reference_frame=ReferenceFrame.VIEWER,
        viewpoint_context_id="vp1",
        execution_policy=ExecutionPolicy.SOFT,
    )
    return GroundingQuery(
        raw_query="facing the door, cabinet on the right",
        root=QueryNode(
            categories=["cabinet"],
            spatial_constraints=[sc],
            node_id="root",
        ),
        expect_unique=True,
        viewpoint_contexts=[ctx],
    )


class TestTrajectoryProvenance(unittest.TestCase):
    def test_pack_under_scene_root_is_preferred(self) -> None:
        with TemporaryDirectory() as tmp:
            scene = Path(tmp) / "sceneXXXX_00"
            scene.mkdir()
            (scene / "traj.txt").write_text("")
            raw = scene.parent / "raw"
            raw.mkdir()
            (raw / "traj.txt").write_text("")
            pack_dir = scene / "pack_nr3d_v9_catalog_first"
            pack_dir.mkdir()
            pack_file = pack_dir / "camera_trajectory.json"
            pack_file.write_text(json.dumps([_identity_pose_list()]))

            selector = _SelectorStub(scene)
            self.assertEqual(selector._find_camera_trajectory_file(), pack_file)

    def test_pack_under_sibling_of_conceptgraph_is_preferred_nr3d_layout(
        self,
    ) -> None:
        # NR3D layout: scene_path = .../scene_XXXX/conceptgraph; packs live
        # at .../scene_XXXX/pack_*.
        with TemporaryDirectory() as tmp:
            scene_root = Path(tmp) / "scene0629_00"
            scene_root.mkdir()
            cg = scene_root / "conceptgraph"
            cg.mkdir()
            (cg / "scene_info.json").write_text("{}")
            raw = scene_root / "raw"
            raw.mkdir()
            (raw / "traj.txt").write_text("")
            pack_dir = scene_root / "pack_nr3d_v9_catalog_first"
            pack_dir.mkdir()
            pack_file = pack_dir / "camera_trajectory.json"
            pack_file.write_text(json.dumps([_identity_pose_list()]))

            selector = _SelectorStub(cg)
            found = selector._find_camera_trajectory_file()
            self.assertEqual(found, pack_file)

    def test_traj_txt_fallback_when_no_pack(self) -> None:
        with TemporaryDirectory() as tmp:
            scene = Path(tmp) / "sceneXXXX_00"
            scene.mkdir()
            cg = scene.parent / "conceptgraph"
            cg.mkdir()
            (cg / "traj.txt").write_text("dummy")

            selector = _SelectorStub(scene)
            found = selector._find_camera_trajectory_file()
            self.assertIsNotNone(found)
            self.assertTrue(str(found).endswith("conceptgraph/traj.txt"))

    def test_enumerate_returns_all_existing_candidates(self) -> None:
        with TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene_x"
            scene.mkdir()
            pack_dir = scene / "pack_v1"
            pack_dir.mkdir()
            pack_file = pack_dir / "camera_trajectory.json"
            pack_file.write_text(json.dumps([_identity_pose_list()]))

            traj = scene / "traj.txt"
            _write_traj_txt_16(traj, 2)

            selector = _SelectorStub(scene)
            files = selector._enumerate_camera_trajectory_files()
            # enumerate canonicalises paths via resolve(); compare resolved.
            self.assertEqual(files[0], pack_file.resolve())
            self.assertIn(traj.resolve(), files)

    def test_no_trajectory_anywhere_returns_none(self) -> None:
        with TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene_x"
            scene.mkdir()
            self.assertIsNone(_SelectorStub(scene)._find_camera_trajectory_file())


class TestJsonPayloadParsing(unittest.TestCase):
    def test_json_list_of_matrices_parses(self) -> None:
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "camera_trajectory.json"
            f.write_text(json.dumps([_identity_pose_list() for _ in range(3)]))
            parsed = KeyframeSelector._read_camera_trajectory_json(f)
            self.assertEqual(len(parsed), 3)
            for p in parsed:
                self.assertEqual(p.shape, (4, 4))

    def test_json_dict_with_frames_key_parses(self) -> None:
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "camera_trajectory.json"
            payload = {
                "frames": [
                    {"world_T_cam": _identity_pose_list()},
                    {"pose": _identity_pose_list()},
                    {"matrix": _identity_pose_list()},
                ]
            }
            f.write_text(json.dumps(payload))
            parsed = KeyframeSelector._read_camera_trajectory_json(f)
            self.assertEqual(len(parsed), 3)

    def test_nr3d_position_only_dict_explicitly_rejected(self) -> None:
        # This is exactly the format observed in
        # data/nr3d/scannet/scene0629_00/pack_nr3d_v9_catalog_first/camera_trajectory.json
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "camera_trajectory.json"
            payload = {
                "0": [0.21, -0.46, 1.88],
                "1": [0.97, -0.49, 1.45],
                "2": [-0.10, 0.55, 1.72],
            }
            f.write_text(json.dumps(payload))
            parsed = KeyframeSelector._read_camera_trajectory_json(f)
            # Explicitly returns [] (not raise) so the loader can fall through
            # to traj.txt. The format detector recognizes the position-only
            # dict and refuses to fabricate orientations.
            self.assertEqual(parsed, [])

    def test_unknown_dict_payload_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "camera_trajectory.json"
            f.write_text(json.dumps({"some_other_key": [1, 2, 3]}))
            with self.assertRaises(ValueError):
                KeyframeSelector._read_camera_trajectory_json(f)


class TestLoadCameraPosesFallthrough(unittest.TestCase):
    def test_nr3d_layout_falls_through_from_pack_to_raw_traj(self) -> None:
        # scene_path = .../scene_xxxx/conceptgraph
        # pack JSON exists but is position-only -> reject
        # raw/traj.txt exists with 16-per-line -> accept
        with TemporaryDirectory() as tmp:
            scene_root = Path(tmp) / "scene0629_00"
            scene_root.mkdir()
            cg = scene_root / "conceptgraph"
            cg.mkdir()
            pack_dir = scene_root / "pack_nr3d_v9_catalog_first"
            pack_dir.mkdir()
            pack_file = pack_dir / "camera_trajectory.json"
            pack_file.write_text(
                json.dumps(
                    {
                        "0": [0.21, -0.46, 1.88],
                        "1": [0.97, -0.49, 1.45],
                    }
                )
            )
            raw = scene_root / "raw"
            raw.mkdir()
            _write_traj_txt_16(raw / "traj.txt", n=5)

            selector = _SelectorStub(cg)
            selector._load_camera_poses()
            self.assertEqual(len(selector.camera_poses), 5)
            self.assertIsNotNone(selector.camera_trajectory_path)
            self.assertTrue(
                str(selector.camera_trajectory_path).endswith("raw/traj.txt")
            )

    def test_no_usable_trajectory_leaves_poses_empty(self) -> None:
        with TemporaryDirectory() as tmp:
            scene_root = Path(tmp) / "scene_y"
            scene_root.mkdir()
            cg = scene_root / "conceptgraph"
            cg.mkdir()
            pack_dir = scene_root / "pack_x"
            pack_dir.mkdir()
            (pack_dir / "camera_trajectory.json").write_text(
                json.dumps({"0": [0.0, 0.0, 0.0]})
            )

            selector = _SelectorStub(cg)
            selector._load_camera_poses()
            self.assertEqual(selector.camera_poses, [])
            self.assertIsNone(selector.camera_trajectory_path)


class TestForceLegacyWorldHard(unittest.TestCase):
    def test_force_demotes_viewer_to_world_hard(self) -> None:
        selector = _SelectorStub(Path("/tmp"))
        stripped = selector._force_legacy_world_hard(_wrap(_viewer_grounding()))
        sc = stripped.hypotheses[0].grounding_query.root.spatial_constraints[0]
        self.assertEqual(sc.reference_frame, ReferenceFrame.WORLD)
        self.assertEqual(sc.execution_policy, ExecutionPolicy.HARD)
        self.assertIsNone(sc.viewpoint_context_id)
        # viewpoint_contexts must also be cleared
        self.assertEqual(stripped.hypotheses[0].grounding_query.viewpoint_contexts, [])

    def test_force_demotes_ambiguous_rank_only_to_world_hard(self) -> None:
        selector = _SelectorStub(Path("/tmp"))
        sc = SpatialConstraint(
            relation="left_of",
            anchors=[QueryNode(categories=["door"], node_id="anchor")],
            reference_frame=ReferenceFrame.AMBIGUOUS,
            viewpoint_context_id=None,
            execution_policy=ExecutionPolicy.RANK_ONLY,
        )
        gq = GroundingQuery(
            raw_query="cabinet to the left",
            root=QueryNode(
                categories=["cabinet"], spatial_constraints=[sc], node_id="root"
            ),
            expect_unique=True,
        )
        forced = selector._force_legacy_world_hard(_wrap(gq))
        out = forced.hypotheses[0].grounding_query.root.spatial_constraints[0]
        self.assertEqual(out.reference_frame, ReferenceFrame.WORLD)
        self.assertEqual(out.execution_policy, ExecutionPolicy.HARD)

    def test_force_idempotent_on_world_hard(self) -> None:
        selector = _SelectorStub(Path("/tmp"))
        sc = SpatialConstraint(
            relation="near",
            anchors=[QueryNode(categories=["door"], node_id="anchor")],
        )
        gq = GroundingQuery(
            raw_query="near door",
            root=QueryNode(
                categories=["cabinet"], spatial_constraints=[sc], node_id="root"
            ),
            expect_unique=True,
        )
        forced = selector._force_legacy_world_hard(_wrap(gq))
        out = forced.hypotheses[0].grounding_query.root.spatial_constraints[0]
        self.assertEqual(out.reference_frame, ReferenceFrame.WORLD)
        self.assertEqual(out.execution_policy, ExecutionPolicy.HARD)


class TestGuardRaisesOnMissingTraj(unittest.TestCase):
    def test_raises_when_viewer_constraint_and_no_traj(self) -> None:
        selector = _SelectorStub(Path("/tmp"))
        with self.assertRaises(RuntimeError) as cm:
            selector._guard_viewpoint_traj_available(_wrap(_viewer_grounding()))
        msg = str(cm.exception)
        # Error message must list searched paths so the operator can fix layout
        self.assertIn("paths tried", msg)
        self.assertIn("traj.txt", msg)
        self.assertIn("camera_trajectory.json", msg)

    def test_silent_when_no_viewer_constraint(self) -> None:
        selector = _SelectorStub(Path("/tmp"))
        sc = SpatialConstraint(
            relation="near",
            anchors=[QueryNode(categories=["door"], node_id="a")],
        )
        gq = GroundingQuery(
            raw_query="near door",
            root=QueryNode(
                categories=["cabinet"], spatial_constraints=[sc], node_id="root"
            ),
            expect_unique=True,
        )
        selector._guard_viewpoint_traj_available(_wrap(gq))

    def test_silent_when_traj_loaded(self) -> None:
        selector = _SelectorStub(Path("/tmp"), camera_poses=[_identity_pose()])
        selector._guard_viewpoint_traj_available(_wrap(_viewer_grounding()))


if __name__ == "__main__":
    unittest.main()

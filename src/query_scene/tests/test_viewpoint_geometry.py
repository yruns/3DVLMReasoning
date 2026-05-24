"""Phase-2 tests: viewer-pose resolution and viewer-frame relation scoring.

Verifies:

- ``viewer_axes`` builds a right-handed (forward, right, up) basis with
  +Z up. ``right`` for forward=+X is -Y (standard convention).
- ``resolve_viewer_pose`` picks a sensible viewer for a single anchor,
  multi-anchor "facing the windows" case, and the geometric fallback
  when no camera trajectory is supplied. Returns None on degenerate input.
- ``viewer_frame_relation_score`` ranks candidates correctly on each
  directional relation in viewer frame.
- End-to-end: the executor under viewer-frame routes ``right_of`` to the
  viewer-frame branch and selects the speaker-right candidate over the
  speaker-left one, even when world X axes disagree with the viewer axes.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from query_scene.core import (
    ExecutionPolicy,
    GroundingQuery,
    QueryNode,
    ReferenceFrame,
    SelectConstraint,
    SpatialConstraint,
    ViewpointContext,
    ViewpointKind,
)
from query_scene.query_executor import QueryExecutor
from query_scene.viewpoint_geometry import (
    ViewerPose,
    resolve_viewer_pose,
    viewer_axes,
    viewer_frame_axis_value,
    viewer_frame_relation_score,
)


def _obj(obj_id: int, category: str, xyz: tuple[float, float, float]) -> object:
    return SimpleNamespace(
        obj_id=obj_id,
        category=category,
        object_tag=category,
        centroid=np.array(xyz, dtype=np.float32),
        bbox_3d=None,
        n_points=10,
        class_name=[category],
        summary=category,
    )


class TestViewerAxes(unittest.TestCase):
    def test_forward_east_has_right_pointing_south(self) -> None:
        f, right, up = viewer_axes(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(f, [1.0, 0.0, 0.0], atol=1e-6)
        # forward (+X) x up (+Z) = -Y; matches "facing east, right hand = south"
        np.testing.assert_allclose(right, [0.0, -1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(up, [0.0, 0.0, 1.0], atol=1e-6)

    def test_forward_north_has_right_pointing_east(self) -> None:
        f, right, _ = viewer_axes(np.array([0.0, 1.0, 0.0]))
        np.testing.assert_allclose(f, [0.0, 1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(right, [1.0, 0.0, 0.0], atol=1e-6)

    def test_pitched_forward_projected_to_horizontal(self) -> None:
        f, right, up = viewer_axes(np.array([1.0, 0.0, 1.0]))
        # Horizontal projection of (1,0,1) is (1,0,0); should normalize to that.
        np.testing.assert_allclose(f, [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(right, [0.0, -1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(up, [0.0, 0.0, 1.0], atol=1e-6)

    def test_vertical_forward_raises(self) -> None:
        with self.assertRaises(ValueError):
            viewer_axes(np.array([0.0, 0.0, 1.0]))


class TestResolveViewerPoseGeometricFallback(unittest.TestCase):
    def test_single_anchor_geometric_pose_points_at_anchor(self) -> None:
        door = _obj(0, "door", (5.0, 0.0, 1.0))
        other = _obj(1, "wall", (-5.0, 0.0, 1.0))
        all_centroids = np.stack([door.centroid, other.centroid]).astype(np.float64)

        pose = resolve_viewer_pose([door], all_object_centroids=all_centroids)

        self.assertIsNotNone(pose)
        self.assertEqual(pose.source, "geometric")
        # Viewer position should sit between room center and the door,
        # standing back 1.5m from the anchor along the room-to-anchor line.
        room_center = all_centroids.mean(axis=0)
        expected_forward = door.centroid - room_center
        expected_forward[2] = 0
        expected_forward /= np.linalg.norm(expected_forward)
        np.testing.assert_allclose(pose.forward, expected_forward, atol=1e-5)

        # Viewer should be looking at the door (delta toward anchor along forward).
        anchor_dir = door.centroid - pose.position
        anchor_dir = anchor_dir / np.linalg.norm(anchor_dir)
        self.assertGreater(float(np.dot(anchor_dir, pose.forward)), 0.95)

    def test_multi_anchor_facing_windows_uses_centroid(self) -> None:
        windows = [_obj(i, "window", (i - 1.0, 5.0, 1.5)) for i in range(3)]
        floor = _obj(99, "floor", (0.0, 0.0, 0.0))
        all_centroids = np.stack([w.centroid for w in windows] + [floor.centroid]).astype(
            np.float64
        )

        pose = resolve_viewer_pose(windows, all_object_centroids=all_centroids)

        self.assertIsNotNone(pose)
        # Window centroid is (0, 5, 1.5); facing direction should be roughly +Y.
        self.assertGreater(pose.forward[1], 0.9)

    def test_empty_anchors_returns_none(self) -> None:
        self.assertIsNone(resolve_viewer_pose([]))

    def test_room_center_coincident_with_anchor_returns_none(self) -> None:
        door = _obj(0, "door", (0.0, 0.0, 0.0))
        all_centroids = door.centroid[None, :].astype(np.float64)
        self.assertIsNone(
            resolve_viewer_pose([door], all_object_centroids=all_centroids)
        )


class TestResolveViewerPoseAllowGeometricFallbackFlag(unittest.TestCase):
    """Strict no-fallback flag: when False, refuse to synthesize geometry.

    Mirrors how QueryExecutor calls resolve_viewer_pose - the executor must
    never silently fabricate a pose from the room centroid.
    """

    def test_no_camera_poses_with_flag_off_returns_none(self) -> None:
        door = _obj(0, "door", (5.0, 0.0, 1.0))
        other = _obj(1, "wall", (-5.0, 0.0, 1.0))
        all_centroids = np.stack([door.centroid, other.centroid]).astype(np.float64)

        pose = resolve_viewer_pose(
            [door],
            all_object_centroids=all_centroids,
            camera_poses=None,
            allow_geometric_fallback=False,
        )
        self.assertIsNone(pose)

    def test_empty_camera_poses_with_flag_off_returns_none(self) -> None:
        door = _obj(0, "door", (5.0, 0.0, 1.0))
        pose = resolve_viewer_pose(
            [door],
            camera_poses=[],
            allow_geometric_fallback=False,
        )
        self.assertIsNone(pose)

    def test_degenerate_trajectory_with_flag_off_returns_none(self) -> None:
        # Only "frames" are nonsense shapes that fail viewer_axes; trajectory
        # mode finds nothing usable, and the geometric fallback is off.
        door = _obj(0, "door", (5.0, 0.0, 1.0))
        bad_pose = np.zeros((4, 4), dtype=np.float64)
        pose = resolve_viewer_pose(
            [door],
            camera_poses=[bad_pose],
            allow_geometric_fallback=False,
        )
        self.assertIsNone(pose)

    def test_flag_default_true_preserves_geometric_fallback(self) -> None:
        door = _obj(0, "door", (5.0, 0.0, 1.0))
        other = _obj(1, "wall", (-5.0, 0.0, 1.0))
        all_centroids = np.stack([door.centroid, other.centroid]).astype(np.float64)

        pose = resolve_viewer_pose(
            [door],
            all_object_centroids=all_centroids,
            camera_poses=None,
        )
        self.assertIsNotNone(pose)
        self.assertEqual(pose.source, "geometric")


class TestResolveViewerPoseTrajectoryMode(unittest.TestCase):
    def test_trajectory_picks_best_aligned_frame(self) -> None:
        door = _obj(0, "door", (10.0, 0.0, 1.0))

        def world_T_cam(pos: tuple[float, float, float], forward_yaw: float) -> np.ndarray:
            # Camera convention: -Z is forward in cam frame.
            # We build world_T_cam where cam's -Z column = world forward direction.
            cx, cy = np.cos(forward_yaw), np.sin(forward_yaw)
            forward = np.array([cx, cy, 0.0])
            # cam +Z in world = -forward.
            cam_z = -forward
            cam_y = np.array([0.0, 0.0, 1.0])
            cam_x = np.cross(cam_y, cam_z)
            cam_x = cam_x / np.linalg.norm(cam_x)
            R = np.stack([cam_x, cam_y, cam_z], axis=1)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = pos
            return T

        # Three frames: one faces +X (toward the door), one faces +Y, one faces -X.
        toward_door = world_T_cam((0.0, 0.0, 1.0), 0.0)        # forward = +X
        toward_y = world_T_cam((0.0, 0.0, 1.0), np.pi / 2)      # forward = +Y
        away = world_T_cam((0.0, 0.0, 1.0), np.pi)              # forward = -X
        poses = [toward_y, away, toward_door]

        pose = resolve_viewer_pose([door], camera_poses=poses)
        self.assertIsNotNone(pose)
        self.assertEqual(pose.source, "trajectory")
        np.testing.assert_allclose(pose.forward, [1.0, 0.0, 0.0], atol=1e-5)


class TestViewerFrameRelationScore(unittest.TestCase):
    def _pose(self, forward: tuple[float, float, float]) -> ViewerPose:
        f, right, up = viewer_axes(np.array(forward, dtype=np.float64))
        return ViewerPose(
            position=np.zeros(3),
            forward=f,
            right=right,
            up=up,
            source="geometric",
        )

    def test_right_of_when_candidate_is_speaker_right_score_positive(self) -> None:
        pose = self._pose((1.0, 0.0, 0.0))  # face +X, right = -Y
        anchor = np.array([5.0, 0.0, 0.0])
        cand_right = np.array([5.0, -2.0, 0.0])  # -Y from anchor = speaker right
        cand_left = np.array([5.0, 2.0, 0.0])
        cand_behind = np.array([3.0, 0.0, 0.0])

        s_right = viewer_frame_relation_score("right_of", cand_right, anchor, pose)
        s_left = viewer_frame_relation_score("right_of", cand_left, anchor, pose)
        s_behind = viewer_frame_relation_score("right_of", cand_behind, anchor, pose)

        self.assertGreater(s_right, 0.0)
        self.assertEqual(s_left, 0.0)
        self.assertEqual(s_behind, 0.0)

    def test_left_of_in_front_of_behind_signs(self) -> None:
        pose = self._pose((1.0, 0.0, 0.0))
        anchor = np.array([5.0, 0.0, 0.0])
        cand_left = np.array([5.0, 2.0, 0.0])    # +Y from anchor
        cand_in_front = np.array([3.0, 0.0, 0.0])  # closer to viewer than anchor
        cand_behind = np.array([7.0, 0.0, 0.0])    # farther than anchor

        self.assertGreater(
            viewer_frame_relation_score("left_of", cand_left, anchor, pose), 0.0
        )
        self.assertEqual(
            viewer_frame_relation_score("left_of", cand_in_front, anchor, pose), 0.0
        )
        self.assertGreater(
            viewer_frame_relation_score("in_front_of", cand_in_front, anchor, pose),
            0.0,
        )
        self.assertGreater(
            viewer_frame_relation_score("behind", cand_behind, anchor, pose), 0.0
        )

    def test_unknown_relation_returns_zero(self) -> None:
        pose = self._pose((1.0, 0.0, 0.0))
        zero = viewer_frame_relation_score(
            "on_top_of", np.array([0, 0, 1]), np.array([0, 0, 0]), pose
        )
        self.assertEqual(zero, 0.0)


class TestViewerFrameAxisValue(unittest.TestCase):
    def test_right_axis_projection_signs(self) -> None:
        pose = ViewerPose(
            position=np.zeros(3),
            forward=np.array([1.0, 0.0, 0.0]),
            right=np.array([0.0, -1.0, 0.0]),
            up=np.array([0.0, 0.0, 1.0]),
            source="test",
        )
        # Candidate at (0, -3, 0): dot with right=(0,-1,0) is +3 (speaker right).
        self.assertEqual(
            viewer_frame_axis_value(np.array([0, -3, 0]), pose, "right"), 3.0
        )
        # Candidate at (0, 3, 0): -3 (speaker left).
        self.assertEqual(
            viewer_frame_axis_value(np.array([0, 3, 0]), pose, "right"), -3.0
        )

    def test_invalid_axis_raises(self) -> None:
        pose = ViewerPose(
            position=np.zeros(3),
            forward=np.array([1.0, 0.0, 0.0]),
            right=np.array([0.0, -1.0, 0.0]),
            up=np.array([0.0, 0.0, 1.0]),
            source="test",
        )
        with self.assertRaises(ValueError):
            viewer_frame_axis_value(np.zeros(3), pose, "diagonal")


def _world_T_cam_facing(
    position: tuple[float, float, float], forward: tuple[float, float, float]
) -> np.ndarray:
    """Build a 4x4 world_T_cam matrix whose camera looks along ``forward``.

    Camera convention follows OpenGL / OpenCV: camera looks along -Z in its
    own frame, +Y up, +X right. So world_T_cam[:3, 2] (cam +Z column in
    world coordinates) equals -forward.
    """
    f = np.asarray(forward, dtype=np.float64)
    f = f / np.linalg.norm(f)
    cam_z_world = -f
    cam_y_world = np.array([0.0, 0.0, 1.0])
    cam_x_world = np.cross(cam_y_world, cam_z_world)
    cam_x_world = cam_x_world / np.linalg.norm(cam_x_world)
    cam_y_world = np.cross(cam_z_world, cam_x_world)
    R = np.stack([cam_x_world, cam_y_world, cam_z_world], axis=1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(position, dtype=np.float64)
    return T


class TestExecutorViewerFrameEndToEnd(unittest.TestCase):
    """Cabinet-on-the-right when facing the door.

    Executor banishes the geometric fallback (allow_geometric_fallback=False
    when calling resolve_viewer_pose). These tests therefore supply a
    synthetic camera trajectory so trajectory mode picks a usable pose.
    """

    def _build_scene(self) -> tuple[list[object], object, list[object]]:
        # Scene layout (top-down, x right, y up):
        #     y
        #     |     [door @ (0, 5)]
        #     |  [cab_LEFT @ (-2,5)]  [cab_RIGHT @ (2,5)]
        #     +----- x ----->
        # The speaker stands roughly at origin facing +Y (toward door).
        # In viewer frame: right = +X (since forward=+Y, right=forward x up = +X).
        door = _obj(0, "door", (0.0, 5.0, 0.0))
        cab_left = _obj(1, "cabinet", (-2.0, 5.0, 0.0))
        cab_right = _obj(2, "cabinet", (2.0, 5.0, 0.0))
        # Filler so room centroid is roughly origin.
        floor = _obj(99, "floor", (0.0, 0.0, 0.0))
        return [door, cab_left, cab_right, floor], door, [cab_left, cab_right]

    def _camera_poses_facing_door(self) -> list[np.ndarray]:
        # A frame standing roughly at origin facing the door, plus a
        # decoy frame facing away so the trajectory-mode picker has to
        # discriminate (and the test verifies it picks the good one).
        return [
            _world_T_cam_facing((0.0, 0.0, 1.2), (0.0, 1.0, 0.0)),  # faces door
            _world_T_cam_facing((0.0, 0.0, 1.2), (0.0, -1.0, 0.0)),  # away
        ]

    def _build_query(self) -> GroundingQuery:
        ctx = ViewpointContext(
            id="vp_door",
            kind=ViewpointKind.FACING_ANCHOR,
            facing_anchor=QueryNode(categories=["door"], node_id="vp_door_facing"),
            raw_phrase="facing the door",
        )
        sc = SpatialConstraint(
            relation="right_of",
            anchors=[QueryNode(categories=["door"], node_id="anchor")],
            reference_frame=ReferenceFrame.VIEWER,
            viewpoint_context_id="vp_door",
            execution_policy=ExecutionPolicy.SOFT,
        )
        return GroundingQuery(
            raw_query="facing the door, the cabinet on the right",
            root=QueryNode(
                categories=["cabinet"],
                spatial_constraints=[sc],
                node_id="root",
            ),
            expect_unique=False,
            viewpoint_contexts=[ctx],
        )

    def test_viewer_frame_right_of_picks_speaker_right_cabinet(self) -> None:
        objs, _door, cabinets = self._build_scene()
        executor = QueryExecutor(
            objects=objs, camera_poses=self._camera_poses_facing_door()
        )

        result = executor.execute(self._build_query())

        ids_sorted_by_score = sorted(
            result.matched_objects, key=lambda o: -result.scores[o.obj_id]
        )
        self.assertEqual(ids_sorted_by_score[0].obj_id, 2)  # cab_right scores higher
        self.assertEqual({o.obj_id for o in cabinets}, {1, 2})

    def test_viewer_frame_select_x_position_picks_speaker_right(self) -> None:
        objs, _door, _ = self._build_scene()
        executor = QueryExecutor(
            objects=objs, camera_poses=self._camera_poses_facing_door()
        )

        ctx = ViewpointContext(
            id="vp_door",
            kind=ViewpointKind.FACING_ANCHOR,
            facing_anchor=QueryNode(categories=["door"], node_id="vp_door_facing"),
            raw_phrase="facing the door",
        )
        sel = SelectConstraint(
            constraint_type="superlative",
            metric="x_position",
            order="max",
            reference=None,
            position=None,
            reference_frame=ReferenceFrame.VIEWER,
            viewpoint_context_id="vp_door",
            execution_policy=ExecutionPolicy.HARD,
        )
        gq = GroundingQuery(
            raw_query="facing the door, the rightmost cabinet",
            root=QueryNode(
                categories=["cabinet"],
                select_constraint=sel,
                node_id="root",
            ),
            expect_unique=True,
            viewpoint_contexts=[ctx],
        )

        result = executor.execute(gq)
        self.assertEqual(len(result.matched_objects), 1)
        self.assertEqual(result.matched_objects[0].obj_id, 2)  # cab_right

    def test_viewer_pose_unresolvable_falls_through_to_policy_default(self) -> None:
        # Set up a constraint asking for a viewer pose whose context anchor
        # category does not exist in scene. The constraint is SOFT, so the
        # executor must keep candidates rather than emptying them. No
        # camera_poses provided - and the executor MUST NOT silently invoke
        # the geometric fallback in resolve_viewer_pose; it must hand back
        # None and let the policy floor decide.
        cabinets = [
            _obj(1, "cabinet", (-2.0, 5.0, 0.0)),
            _obj(2, "cabinet", (2.0, 5.0, 0.0)),
        ]
        executor = QueryExecutor(objects=cabinets)

        ctx = ViewpointContext(
            id="vp_missing",
            kind=ViewpointKind.FACING_ANCHOR,
            facing_anchor=QueryNode(
                categories=["UNKNOW"], node_id="vp_missing_facing"
            ),
            raw_phrase="facing the mystery",
        )
        sc = SpatialConstraint(
            relation="right_of",
            anchors=[QueryNode(categories=["cabinet"], node_id="anchor")],
            reference_frame=ReferenceFrame.VIEWER,
            viewpoint_context_id="vp_missing",
            execution_policy=ExecutionPolicy.SOFT,
        )
        gq = GroundingQuery(
            raw_query="facing mystery, cabinet on the right",
            root=QueryNode(
                categories=["cabinet"],
                spatial_constraints=[sc],
                node_id="root",
            ),
            expect_unique=False,
            viewpoint_contexts=[ctx],
        )

        result = executor.execute(gq)
        # SOFT policy must keep all candidates even though the viewer pose
        # could not be computed.
        self.assertEqual({o.obj_id for o in result.matched_objects}, {1, 2})

    def test_executor_with_anchorable_context_but_no_camera_poses_returns_neutral(
        self,
    ) -> None:
        # The facing anchor is a real scene category (door), so the previous
        # path would have synthesized a geometric pose from the room centroid.
        # With allow_geometric_fallback=False in the executor, no
        # camera_poses means no viewer pose - so HARD must return empty,
        # never silently world-frame the relation.
        objs, _door, _cabinets = self._build_scene()
        executor = QueryExecutor(objects=objs)  # no camera_poses

        ctx = ViewpointContext(
            id="vp_door",
            kind=ViewpointKind.FACING_ANCHOR,
            facing_anchor=QueryNode(categories=["door"], node_id="vp_door_facing"),
            raw_phrase="facing the door",
        )
        sc = SpatialConstraint(
            relation="right_of",
            anchors=[QueryNode(categories=["door"], node_id="anchor")],
            reference_frame=ReferenceFrame.VIEWER,
            viewpoint_context_id="vp_door",
            execution_policy=ExecutionPolicy.HARD,
        )
        gq = GroundingQuery(
            raw_query="facing the door, cabinet on the right",
            root=QueryNode(
                categories=["cabinet"],
                spatial_constraints=[sc],
                node_id="root",
            ),
            expect_unique=False,
            viewpoint_contexts=[ctx],
        )

        result = executor.execute(gq)
        # No trajectory, no geometric fallback in executor path, HARD => empty.
        self.assertEqual(result.matched_objects, [])


class TestOrdinalViewerFrameAxes(unittest.TestCase):
    """Regression: ordinal x_position / y_position must use viewer-frame
    projection when reference_frame=VIEWER, not world coordinates.

    Constructed so world-X order disagrees with viewer-right order, so a
    fall-through to ``pos[0]`` would pick a different window than the
    viewer-frame branch.
    """

    def _build_scene(self) -> tuple[list[object], list[object]]:
        # Speaker stands at origin facing +X (toward a far wall at x=6).
        # viewer right = +X x +Z = -Y, so positive viewer-right corresponds
        # to -Y in world coordinates.
        #
        # Windows laid out so:
        #   - world-X order ascending:  w1 (x=4), w2 (x=5), w3 (x=6)
        #   - viewer-right ascending:   w3 (-5), w1 (0),    w2 (+5)
        # ordinal position=1 with metric=x_position, order=asc
        #   - world frame would pick w1 (smallest world X)
        #   - viewer frame must pick w3 (most-left from speaker's POV)
        w1 = _obj(1, "window", (4.0, 0.0, 1.5))
        w2 = _obj(2, "window", (5.0, -5.0, 1.5))
        w3 = _obj(3, "window", (6.0, 5.0, 1.5))
        floor = _obj(99, "floor", (0.0, 0.0, 0.0))
        return [w1, w2, w3, floor], [w1, w2, w3]

    def _camera_poses_facing_pos_x(self) -> list[np.ndarray]:
        return [_world_T_cam_facing((0.0, 0.0, 1.5), (1.0, 0.0, 0.0))]

    def _build_query(self, *, frame: ReferenceFrame) -> GroundingQuery:
        ctx = ViewpointContext(
            id="vp_wall",
            kind=ViewpointKind.FACING_ANCHOR,
            facing_anchor=QueryNode(categories=["window"], node_id="vp_facing"),
            raw_phrase="facing the wall of windows",
        )
        sel_kwargs = {
            "constraint_type": "ordinal",
            "metric": "x_position",
            "order": "asc",
            "reference": None,
            "position": 1,
            "execution_policy": ExecutionPolicy.HARD,
        }
        if frame == ReferenceFrame.VIEWER:
            sel_kwargs.update(
                reference_frame=ReferenceFrame.VIEWER,
                viewpoint_context_id="vp_wall",
            )
        sel = SelectConstraint(**sel_kwargs)
        return GroundingQuery(
            raw_query="facing the wall of windows, the first window from the left",
            root=QueryNode(
                categories=["window"],
                select_constraint=sel,
                node_id="root",
            ),
            expect_unique=True,
            viewpoint_contexts=[ctx],
        )

    def test_ordinal_world_frame_picks_smallest_world_x(self) -> None:
        objs, _windows = self._build_scene()
        executor = QueryExecutor(
            objects=objs, camera_poses=self._camera_poses_facing_pos_x()
        )
        # Baseline: ordinal in world frame uses pos[0] -> obj_id=1
        result = executor.execute(self._build_query(frame=ReferenceFrame.WORLD))
        self.assertEqual(len(result.matched_objects), 1)
        self.assertEqual(result.matched_objects[0].obj_id, 1)

    def test_ordinal_viewer_frame_picks_speaker_leftmost(self) -> None:
        objs, _windows = self._build_scene()
        executor = QueryExecutor(
            objects=objs, camera_poses=self._camera_poses_facing_pos_x()
        )
        # With reference_frame=VIEWER, sort key is viewer-right projection.
        # asc order picks the most-negative viewer-right => speaker-leftmost.
        # World order would pick obj_id=1; viewer order picks obj_id=3.
        result = executor.execute(self._build_query(frame=ReferenceFrame.VIEWER))
        self.assertEqual(len(result.matched_objects), 1)
        self.assertEqual(result.matched_objects[0].obj_id, 3)


class TestFacingWindowsSelectFixture(unittest.TestCase):
    """Diagnostic row 2: 'Facing the windows, the window on the right side.'.

    Reproduces the world-frame failure mode: in world coords, max(x_position)
    might pick the wrong window. In viewer frame (facing +Y), max along
    right=+X actually does pick the speaker-right window.
    """

    def test_select_max_x_position_picks_speaker_right_window(self) -> None:
        # Windows arrayed along the wall at y=5, speaker standing at origin.
        # Top-down: w_far_left=(-3,5), w_mid=(0,5), w_far_right=(3,5).
        # Executor bans geometric fallback now; supply a synthetic camera
        # pose that faces +Y so trajectory mode resolves the viewer pose.
        windows = [
            _obj(1, "window", (-3.0, 5.0, 1.5)),
            _obj(2, "window", (0.0, 5.0, 1.5)),
            _obj(3, "window", (3.0, 5.0, 1.5)),
        ]
        floor = _obj(99, "floor", (0.0, 0.0, 0.0))
        executor = QueryExecutor(
            objects=windows + [floor],
            camera_poses=[
                _world_T_cam_facing((0.0, 0.0, 1.5), (0.0, 1.0, 0.0))
            ],
        )

        ctx = ViewpointContext(
            id="vp_windows",
            kind=ViewpointKind.FACING_ANCHOR_SET,
            facing_anchor=QueryNode(categories=["window"], node_id="vp_facing"),
            raw_phrase="Facing the windows",
        )
        sel = SelectConstraint(
            constraint_type="superlative",
            metric="x_position",
            order="max",
            reference=None,
            position=None,
            reference_frame=ReferenceFrame.VIEWER,
            viewpoint_context_id="vp_windows",
            execution_policy=ExecutionPolicy.HARD,
        )
        gq = GroundingQuery(
            raw_query="Facing the windows, the window on the right side.",
            root=QueryNode(
                categories=["window"],
                select_constraint=sel,
                node_id="root",
            ),
            expect_unique=True,
            viewpoint_contexts=[ctx],
        )

        result = executor.execute(gq)
        self.assertEqual(len(result.matched_objects), 1)
        # Speaker faces +Y; right = +X. max(right_projection) picks w_far_right (3,5).
        self.assertEqual(result.matched_objects[0].obj_id, 3)


if __name__ == "__main__":
    unittest.main()

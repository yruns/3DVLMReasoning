"""v9 scene BEV builder: mesh + camera trajectory + per-proposal `#id label`.

Per spec Section B. Each benchmark gets its own subclass that knows how to
discover mesh / trajectory / intrinsic paths for its data layout.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from agents.catalog import SceneProposal


@dataclass(frozen=True)
class SceneBEVConfig:
    image_size: int = 1500
    perspective: bool = True
    camera_fov: float = 100.0
    ceiling_normal_threshold: float = -0.6
    label_color_highlight: tuple[int, int, int] = (255, 64, 64)
    label_color_default: tuple[int, int, int] = (32, 32, 32)
    label_bg_highlight: tuple[int, int, int] = (255, 255, 0)
    label_bg_default: tuple[int, int, int] = (255, 255, 255)
    trajectory_color: tuple[int, int, int] = (32, 96, 220)
    trajectory_thickness: int = 3
    proposal_marker_radius: int = 4


class ScanNetSceneBEVBuilderBase(ABC):
    """Shared mesh + trajectory + per-proposal label overlay for ScanNet-based benchmarks."""

    benchmark: str = "scannet"

    def __init__(self, config: SceneBEVConfig | None = None) -> None:
        self.config = config or SceneBEVConfig()

    @abstractmethod
    def resolve_paths(
        self,
        scene_id: str,
        data_root: Path,
    ) -> tuple[Path, Path, Path]:
        """Return (mesh_path, traj_path, intrinsic_path) for the scene."""
        raise NotImplementedError

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self.config), sort_keys=True)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]

    def build_with_labels(
        self,
        *,
        scene_id: str,
        data_root: Path,
        proposals: list[SceneProposal],
        output_path: Path,
        highlight_ids: list[int] | None = None,
    ) -> Path:
        mesh_path, traj_path, intr_path = self.resolve_paths(scene_id, data_root)
        for p in (mesh_path, traj_path, intr_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"required asset missing for scene_id={scene_id}: {p}"
                )

        img, scene_bounds = self._render_mesh_with_traj(mesh_path, traj_path, intr_path)
        img = self._overlay_proposal_labels(
            img, proposals, scene_bounds=scene_bounds, highlight_ids=highlight_ids
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        logger.info(f"[scene_bev] wrote {output_path}")
        return output_path

    def _render_mesh_with_traj(
        self,
        mesh_path: Path,
        traj_path: Path,
        intr_path: Path,
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        """Render BEV using the same pipeline as the legacy OpenEQA builder.

        Returns (rgb_image_uint8, scene_bounds=(xmin, ymin, xmax, ymax)).
        """
        import open3d as o3d  # heavy import: lazy

        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.compute_triangle_normals()
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        tris = np.asarray(mesh.triangles)
        colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
        normals = np.asarray(mesh.triangle_normals)

        K = np.loadtxt(str(intr_path))[:3, :3]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        traj = np.loadtxt(str(traj_path)).reshape(-1, 4, 4)
        cam_positions = traj[:, :3, 3]

        from query_scene.bev_builder import OpenEQAScanNetBEVBuilder

        legacy = OpenEQAScanNetBEVBuilder()
        visible = legacy._compute_frustum_visibility(
            verts, traj, fx, fy, cx, cy, 1296, 968
        )
        tri_visible = visible[tris[:, 0]] & visible[tris[:, 1]] & visible[tris[:, 2]]
        facing_down = normals[:, 2] < self.config.ceiling_normal_threshold
        tri_keep = tri_visible & (~facing_down)
        kept = tris[tri_keep]
        if len(kept) == 0:
            kept = tris[tri_visible]
        img = legacy._render_perspective(verts, colors, kept, cam_positions)
        img = legacy._draw_trajectory(img, verts, cam_positions)
        xs, ys = verts[:, 0], verts[:, 1]
        bounds = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        return img, bounds

    def _overlay_proposal_labels(
        self,
        img: np.ndarray,
        proposals: list[SceneProposal],
        scene_bounds: tuple[float, float, float, float],
        highlight_ids: list[int] | None,
    ) -> np.ndarray:
        img = img.copy()
        h, w = img.shape[:2]
        xmin, ymin, xmax, ymax = scene_bounds
        span_x = max(xmax - xmin, 1e-6)
        span_y = max(ymax - ymin, 1e-6)
        highlight_set: set[int] = set(highlight_ids) if highlight_ids is not None else set()
        for proposal in proposals:
            if highlight_ids is not None and proposal.proposal_id not in highlight_set:
                continue
            x_world, y_world = proposal.position_3d[0], proposal.position_3d[1]
            u = int((x_world - xmin) / span_x * (w - 1))
            v = int((1.0 - (y_world - ymin) / span_y) * (h - 1))
            u = max(0, min(w - 1, u))
            v = max(0, min(h - 1, v))
            highlighted = (
                highlight_ids is not None
                and proposal.proposal_id in highlight_set
            )
            color = (
                self.config.label_color_highlight
                if highlighted
                else self.config.label_color_default
            )
            bg = (
                self.config.label_bg_highlight
                if highlighted
                else self.config.label_bg_default
            )
            cv2.circle(img, (u, v), self.config.proposal_marker_radius, color, -1)
            label = f"#{proposal.proposal_id} {proposal.category}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.45
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
            text_org = (u + 6, v - 6)
            bg_x1 = text_org[0] - 2
            bg_y1 = text_org[1] - th - 2
            bg_x2 = text_org[0] + tw + 2
            bg_y2 = text_org[1] + baseline
            cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), bg, -1)
            cv2.putText(img, label, text_org, font, scale, color, thickness, cv2.LINE_AA)
        return img


__all__ = ["SceneBEVConfig", "ScanNetSceneBEVBuilderBase"]

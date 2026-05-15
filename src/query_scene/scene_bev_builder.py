"""v9 scene BEV builder: mesh + camera trajectory + per-proposal `#id label`.

Per spec Section B. Each benchmark gets its own subclass that knows how to
discover mesh / trajectory / intrinsic paths for its data layout.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from agents.catalog import SceneProposal


def _crop_to_non_white(
    img: np.ndarray, *, margin: int = 8, threshold: int = 250
) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop an image to its non-white bounding box plus a fixed pixel margin.

    Returns (cropped_image, (offset_x, offset_y)). Offsets allow callers to
    translate any pre-computed (u, v) image coordinates into the cropped frame.
    """
    non_white = np.any(img < threshold, axis=2)
    if not non_white.any():
        return img, (0, 0)
    rows = np.where(non_white.any(axis=1))[0]
    cols = np.where(non_white.any(axis=0))[0]
    y_min = max(0, int(rows.min()) - margin)
    y_max = min(img.shape[0], int(rows.max()) + 1 + margin)
    x_min = max(0, int(cols.min()) - margin)
    x_max = min(img.shape[1], int(cols.max()) + 1 + margin)
    return img[y_min:y_max, x_min:x_max], (x_min, y_min)


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
    crop_margin: int = 8
    label_font_scale: float = 0.85
    label_font_thickness: int = 2


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

    def resolve_axis_align(
        self,
        scene_id: str,
        data_root: Path,
    ) -> np.ndarray | None:
        """Return the 4x4 axis-alignment matrix for the scene, or None.

        ScanNet stores meshes in the raw scan frame while trajectory poses,
        Mask3D / GT proposal bboxes, and ConceptGraph objects all live in the
        axis-aligned frame. Rendering them together requires applying this
        matrix to the mesh vertices first. Subclasses override when they
        know where to find it; the base implementation looks for
        ``axisAlignment`` lines in standard ScanNet ``<scene>.txt`` metadata
        under common layouts.
        """
        candidates = [
            data_root / scene_id / "raw" / f"{scene_id}.txt",
            data_root.parent / "scannet_aux" / scene_id / f"{scene_id}.txt",
            data_root / scene_id / f"{scene_id}.txt",
        ]
        for path in candidates:
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line.startswith("axisAlignment"):
                    continue
                values = line.split("=", 1)[1].strip().split()
                if len(values) != 16:
                    continue
                return np.array(values, dtype=np.float64).reshape(4, 4)
        return None

    def config_hash(self) -> str:
        """Hash of the static SceneBEVConfig only (no proposals / highlights)."""
        payload = json.dumps(asdict(self.config), sort_keys=True)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]

    def render_hash(
        self,
        proposals: list[SceneProposal],
        highlight_ids: list[int] | None,
    ) -> str:
        """Stable hash of (config, proposals, highlights, benchmark).

        Two BEV calls that produce a byte-identical image agree on this hash;
        any change to SceneBEVConfig fields, the proposal pool that drives the
        label overlay, the highlight subset, or the benchmark builder class
        invalidates it. 12-char hex is short enough for readable filenames and
        wide enough to make accidental collisions effectively impossible.
        """
        prop_signature = sorted(
            (
                int(p.proposal_id),
                str(p.category),
                tuple(round(c, 4) for c in p.position_3d),
            )
            for p in proposals
        )
        payload = json.dumps(
            {
                "benchmark": self.benchmark,
                "config": asdict(self.config),
                "proposals": prop_signature,
                "highlights": (
                    sorted(int(i) for i in highlight_ids)
                    if highlight_ids is not None
                    else None
                ),
            },
            sort_keys=True,
            default=list,
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]

    def scene_cache_dir(self, scene_id: str, data_root: Path) -> Path:
        """Per-scene BEV cache directory: <data_root>/<scene_id>/bev_cache/."""
        return data_root / scene_id / "bev_cache"

    def cache_path(
        self,
        *,
        scene_id: str,
        data_root: Path,
        proposals: list[SceneProposal],
        highlight_ids: list[int] | None,
    ) -> Path:
        """Resolve the on-disk cache filename for a (scene, config, proposals,
        highlights) tuple. Filename format: ``scene_bev_<render_hash>.png``."""
        digest = self.render_hash(proposals, highlight_ids)
        return self.scene_cache_dir(scene_id, data_root) / f"scene_bev_{digest}.png"

    def build_with_labels(
        self,
        *,
        scene_id: str,
        data_root: Path,
        proposals: list[SceneProposal],
        output_path: Path,
        highlight_ids: list[int] | None = None,
        use_cache: bool = True,
    ) -> Path:
        """Render the BEV (or return a cached copy under <scene>/bev_cache/).

        When ``use_cache`` is True (default), the renderer first checks the
        per-scene cache directory for an image whose filename matches the
        current (config, proposals, highlights) tuple. If a hit is found, that
        path is copied to ``output_path`` and returned. Otherwise the BEV is
        rendered fresh and written to BOTH ``output_path`` and the cache slot
        so the next caller in this scene benefits from the warm cache.
        """
        cache_target = self.cache_path(
            scene_id=scene_id,
            data_root=data_root,
            proposals=proposals,
            highlight_ids=highlight_ids,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if use_cache and cache_target.exists():
            if cache_target.resolve() != output_path.resolve():
                output_path.write_bytes(cache_target.read_bytes())
            logger.info(
                f"[scene_bev] cache hit for scene_id={scene_id}: {cache_target.name}"
            )
            return output_path

        mesh_path, traj_path, intr_path = self.resolve_paths(scene_id, data_root)
        for p in (mesh_path, traj_path, intr_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"required asset missing for scene_id={scene_id}: {p}"
                )

        axis_align = self.resolve_axis_align(scene_id, data_root)
        img, scene_bounds = self._render_mesh_with_traj(
            mesh_path, traj_path, intr_path, axis_align=axis_align,
        )
        img, (crop_ox, crop_oy) = _crop_to_non_white(
            img, margin=self.config.crop_margin
        )
        if isinstance(scene_bounds, dict):
            scene_bounds["crop_offset"] = (crop_ox, crop_oy)
        img = self._overlay_proposal_labels(
            img, proposals, scene_bounds=scene_bounds, highlight_ids=highlight_ids
        )
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), bgr)
        if use_cache:
            cache_target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cache_target), bgr)
        logger.info(
            f"[scene_bev] rendered scene_id={scene_id} -> {output_path}"
            f"{' + cached ' + cache_target.name if use_cache else ''}"
        )
        return output_path

    def _render_mesh_with_traj(
        self,
        mesh_path: Path,
        traj_path: Path,
        intr_path: Path,
        *,
        axis_align: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Render BEV using the legacy OpenEQA builder's perspective camera.

        When ``axis_align`` (4x4) is provided, mesh vertices and triangle
        normals are transformed into the axis-aligned frame BEFORE rendering
        so they sit in the same coordinate space as the camera trajectory and
        the proposal centroids. Without this, the ScanNet raw mesh frame is
        offset/rotated from the aligned proposal frame and the rendered mesh
        will not overlap the label markers.

        Returns ``(rgb_image_uint8, view_params)``. ``view_params`` carries the
        legacy builder's cached camera (R, t, f, c) so the v9 label overlay
        projects through the SAME camera that drew the mesh + trajectory.
        """
        import open3d as o3d  # heavy import: lazy

        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.compute_triangle_normals()
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        tris = np.asarray(mesh.triangles)
        colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
        normals = np.asarray(mesh.triangle_normals)

        if axis_align is not None:
            # Mesh -> aligned frame so the trajectory + proposals (already in
            # aligned coords) overlay correctly.
            verts_h = np.concatenate(
                [verts, np.ones((verts.shape[0], 1), dtype=np.float64)], axis=1,
            )
            verts = (axis_align @ verts_h.T).T[:, :3]
            # Rotate triangle normals (no translation) so the ceiling/floor
            # discrimination still uses the world-up axis.
            normals = (axis_align[:3, :3] @ normals.T).T

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
        view = {
            "R": legacy._view_R,
            "t": legacy._view_t,
            "f": legacy._view_f,
            "c": legacy._view_c,
            "image_size": img.shape[1],
        }
        return img, view

    def _project_centroid(
        self,
        position_3d: tuple[float, float, float],
        view_or_bounds: dict | tuple[float, float, float, float],
        img_shape: tuple[int, int],
    ) -> tuple[int, int] | None:
        """Project a world-space proposal centroid to image (u, v).

        Two projection modes:

        - ``view_or_bounds`` is a dict with ``R, t, f, c`` (perspective camera
          shared with mesh + trajectory): project through that camera so the
          label lands on the rendered proposal.
        - ``view_or_bounds`` is a 4-tuple (xmin, ymin, xmax, ymax): fall back
          to a simple xy-bbox linear map. Kept so unit tests that don't
          spin up the open3d/perspective stack still work.

        Returns ``None`` when the proposal is behind the camera or outside
        the image plane.
        """
        h, w = img_shape[:2]
        if isinstance(view_or_bounds, dict):
            R = np.asarray(view_or_bounds["R"], dtype=np.float64)
            t = np.asarray(view_or_bounds["t"], dtype=np.float64)
            f = float(view_or_bounds["f"])
            c = float(view_or_bounds["c"])
            full_hw = int(view_or_bounds["image_size"])
            h_chk, w_chk = full_hw, full_hw
            world = np.asarray(position_3d, dtype=np.float64)
            cam = R @ world + t
            if cam[2] < 0.01:
                return None
            u = f * cam[0] / cam[2] + c
            v = f * cam[1] / cam[2] + c
            if (
                not np.isfinite(u)
                or not np.isfinite(v)
                or u < -10
                or u > w_chk + 10
                or v < -10
                or v > h_chk + 10
            ):
                return None
            crop_offset = view_or_bounds.get("crop_offset", (0, 0))
            u_out = int(u) - int(crop_offset[0])
            v_out = int(v) - int(crop_offset[1])
            return u_out, v_out
        xmin, ymin, xmax, ymax = view_or_bounds
        span_x = max(xmax - xmin, 1e-6)
        span_y = max(ymax - ymin, 1e-6)
        x_world, y_world = position_3d[0], position_3d[1]
        u = int((x_world - xmin) / span_x * (w - 1))
        v = int((1.0 - (y_world - ymin) / span_y) * (h - 1))
        return max(0, min(w - 1, u)), max(0, min(h - 1, v))

    def _overlay_proposal_labels(
        self,
        img: np.ndarray,
        proposals: list[SceneProposal],
        scene_bounds: dict | tuple[float, float, float, float],
        highlight_ids: list[int] | None,
    ) -> np.ndarray:
        img = img.copy()
        highlight_set: set[int] = (
            set(highlight_ids) if highlight_ids is not None else set()
        )
        for proposal in proposals:
            if highlight_ids is not None and proposal.proposal_id not in highlight_set:
                continue
            uv = self._project_centroid(proposal.position_3d, scene_bounds, img.shape)
            if uv is None:
                continue
            u, v = uv
            highlighted = (
                highlight_ids is not None
                and proposal.proposal_id in highlight_set
            )
            color = (
                self.config.label_color_highlight
                if highlighted
                else self.config.label_color_default
            )
            cv2.circle(img, (u, v), self.config.proposal_marker_radius, color, -1)
            label = f"#{proposal.proposal_id} {proposal.category}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = self.config.label_font_scale
            thickness = self.config.label_font_thickness
            (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
            text_org = (u + 6, v - 6)
            bg = (
                text_org[0] - 4,
                text_org[1] - th - 4,
                text_org[0] + tw + 4,
                text_org[1] + baseline + 4,
            )
            cv2.rectangle(
                img,
                (bg[0], bg[1]),
                (bg[2], bg[3]),
                self.config.label_bg_highlight if highlighted else self.config.label_bg_default,
                -1,
            )
            cv2.putText(
                img, label, text_org, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA
            )
            cv2.putText(
                img,
                label,
                text_org,
                font,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
        return img


def _scannet_data_root() -> Path:
    root = os.environ.get("SCANNET_DATA_ROOT", "data/scannetv2")
    return Path(root)


def _scannet_search_roots() -> list[Path]:
    """Return candidate roots for ScanNet meshes.

    Honors $SCANNET_DATA_ROOT first, then falls back to the prepared NR3D
    aux mesh tree (``data/nr3d/scannet_aux_meshes``) and the raw ScanNet
    scans tree (``data/ScanNet/scans``) which both ship the canonical
    ``<scene>/<scene>_vh_clean*.ply`` layout on this checkout.
    """
    roots: list[Path] = [_scannet_data_root()]
    extra_env = os.environ.get("SCANNET_DATA_ROOT_EXTRA")
    if extra_env:
        for token in extra_env.split(":"):
            token = token.strip()
            if token:
                roots.append(Path(token))
    for default in (
        "data/nr3d/scannet_aux_meshes",
        "data/ScanNet/scans",
    ):
        path = Path(default)
        if path not in roots:
            roots.append(path)
    return roots


def _find_scannet_mesh(scannet_root: Path, scene_id: str) -> Path:
    """Locate the ScanNet mesh for ``scene_id``.

    Searches ``scannet_root`` first (to honor SCANNET_DATA_ROOT), then the
    additional fallback roots from ``_scannet_search_roots``. The function
    keeps a stable error message so existing callers/tests still recognise
    the canonical "expected <scene>_vh_clean.ply" suffix.
    """
    searched: list[Path] = []
    candidates = [scannet_root]
    for root in _scannet_search_roots():
        if root not in candidates:
            candidates.append(root)
    for root in candidates:
        for filename in (f"{scene_id}_vh_clean.ply", f"{scene_id}_vh_clean_2.ply"):
            candidate = root / scene_id / filename
            searched.append(candidate)
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"scannet mesh for {scene_id} not under {scannet_root}; "
        f"expected {scene_id}_vh_clean.ply (searched: "
        f"{', '.join(str(p) for p in searched)})"
    )


def _resolve_traj(scene_dir: Path) -> Path:
    """Return the trajectory file for a scene, supporting both conceptgraph/
    (canonical Phase-8 layout) and raw/ (legacy prepared-scene layout)."""
    for candidate in (
        scene_dir / "conceptgraph" / "traj.txt",
        scene_dir / "raw" / "traj.txt",
    ):
        if candidate.exists():
            return candidate
    # Returning the canonical path triggers a clean FileNotFoundError in
    # build_with_labels's existence check, with a path the caller recognises.
    return scene_dir / "conceptgraph" / "traj.txt"


class Nr3dScanNetBEVBuilder(ScanNetSceneBEVBuilderBase):
    benchmark = "nr3d"

    def resolve_paths(self, scene_id: str, data_root: Path) -> tuple[Path, Path, Path]:
        scene_dir = data_root / scene_id
        traj = _resolve_traj(scene_dir)
        intr = scene_dir / "raw" / "intrinsic_color.txt"
        mesh = _find_scannet_mesh(_scannet_data_root(), scene_id)
        return mesh, traj, intr


class ScanReferScanNetBEVBuilder(ScanNetSceneBEVBuilderBase):
    benchmark = "scanrefer"

    def resolve_paths(self, scene_id: str, data_root: Path) -> tuple[Path, Path, Path]:
        scene_dir = data_root / scene_id
        traj = _resolve_traj(scene_dir)
        intr_candidates = (
            scene_dir / "conceptgraph" / "intrinsic_color.txt",
            scene_dir / "raw" / "intrinsic_color.txt",
        )
        intr = next((c for c in intr_candidates if c.exists()), intr_candidates[0])
        mesh = _find_scannet_mesh(_scannet_data_root(), scene_id)
        return mesh, traj, intr


def _extract_scannet_scan_id(clip_name: str, cg_dir: Path) -> str:
    info_path = cg_dir / "scene_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            for key in ("scan_id", "scene_id"):
                value = info.get(key)
                if isinstance(value, str) and value.startswith("scene"):
                    return value
        except json.JSONDecodeError:
            pass
    if "scene" in clip_name:
        return "scene" + clip_name.split("-scene", 1)[-1]
    raise ValueError(
        f"could not extract scannet scan_id from clip name {clip_name!r}"
    )


class OpenEqaScanNetBEVBuilder(ScanNetSceneBEVBuilderBase):
    benchmark = "openeqa"

    def resolve_paths(self, scene_id: str, data_root: Path) -> tuple[Path, Path, Path]:
        clip_dir = data_root / scene_id
        cg = clip_dir / "conceptgraph"
        traj = _resolve_traj(clip_dir)
        intr = clip_dir / "raw" / "intrinsic_color.txt"
        if not intr.exists():
            intr = cg / "intrinsic_color.txt"
        scan_id = _extract_scannet_scan_id(scene_id, cg)
        mesh = _find_scannet_mesh(_scannet_data_root(), scan_id)
        return mesh, traj, intr


class Sqa3dScanNetBEVBuilder(ScanNetSceneBEVBuilderBase):
    benchmark = "sqa3d"

    def resolve_paths(self, scene_id: str, data_root: Path) -> tuple[Path, Path, Path]:
        scene_dir = data_root / scene_id
        traj = _resolve_traj(scene_dir)
        intr = scene_dir / "raw" / "intrinsic_color.txt"
        if not intr.exists():
            intr = scene_dir / "conceptgraph" / "intrinsic_color.txt"
        mesh = _find_scannet_mesh(_scannet_data_root(), scene_id)
        return mesh, traj, intr


__all__ = ["SceneBEVConfig", "ScanNetSceneBEVBuilderBase"]
__all__ += ["Nr3dScanNetBEVBuilder"]
__all__ += ["ScanReferScanNetBEVBuilder"]
__all__ += ["OpenEqaScanNetBEVBuilder"]
__all__ += ["Sqa3dScanNetBEVBuilder"]

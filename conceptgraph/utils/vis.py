from __future__ import annotations

import copy
import dataclasses
from collections.abc import Iterable
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import supervision as sv
import torch
from PIL import Image
from supervision.draw.color import Color, ColorPalette

from conceptgraph.slam.models import MapObjectList


class OnlineObjectRenderer:
    """Incremental Open3D visualiser for map objects and camera poses."""

    def __init__(
        self,
        view_param: str | dict,
        base_objects: MapObjectList | None = None,
        gray_map: bool = False,
    ) -> None:
        if base_objects is not None:
            self.n_base_objects = len(base_objects)
            self.base_pcds_vis = copy.deepcopy(base_objects.get_values("pcd"))
            self.base_bboxes_vis = copy.deepcopy(base_objects.get_values("bbox"))
            for i in range(self.n_base_objects):
                self.base_pcds_vis[i] = self.base_pcds_vis[i].voxel_down_sample(
                    voxel_size=0.08
                )
                if gray_map:
                    self.base_pcds_vis[i].paint_uniform_color([0.5, 0.5, 0.5])
            for i in range(self.n_base_objects):
                self.base_bboxes_vis[i].color = [0.5, 0.5, 0.5]
        else:
            self.n_base_objects = 0
            self.base_pcds_vis = []
            self.base_bboxes_vis = []

        self.est_traj: list[np.ndarray] = []
        self.gt_traj: list[np.ndarray] = []
        self.cmap = matplotlib.colormaps.get_cmap("turbo")

        if isinstance(view_param, str):
            self.view_param = o3d.io.read_pinhole_camera_parameters(view_param)
        else:
            self.view_param = view_param

        self.window_height = self.view_param.intrinsic.height
        self.window_width = self.view_param.intrinsic.width

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=self.window_width, height=self.window_height)
        self.vis_ctrl = self.vis.get_view_control()
        self.vis_ctrl.convert_from_pinhole_camera_parameters(self.view_param)

    def filter_base_by_mask(self, mask: Iterable[bool]) -> None:
        """Keep only the base objects where *mask* is True."""
        assert len(list(mask)) == self.n_base_objects
        self.base_pcds_vis = [p for p, m in zip(self.base_pcds_vis, mask) if m]
        self.base_bboxes_vis = [b for b, m in zip(self.base_bboxes_vis, mask) if m]
        self.n_base_objects = len(self.base_pcds_vis)

    def step(
        self,
        image: Image.Image,
        pcds: list[o3d.geometry.PointCloud] | None = None,
        pcd_colors: np.ndarray | None = None,
        est_pose: np.ndarray | None = None,
        gt_pose: np.ndarray | None = None,
        base_objects_color: dict | None = None,
        new_objects: MapObjectList | None = None,
        paint_new_objects: bool = True,
        return_vis_handle: bool = False,
    ) -> tuple[np.ndarray, o3d.visualization.Visualizer | None]:
        """Render one frame and return the captured image."""
        self.vis.clear_geometries()

        if est_pose is not None:
            self.est_traj.append(est_pose)
            frustum = better_camera_frustum(
                est_pose,
                image.height,
                image.width,
                scale=0.5,
                color=[1.0, 0, 0],
            )
            self.vis.add_geometry(frustum)
            if len(self.est_traj) > 1:
                ls = poses2lineset(np.stack(self.est_traj), color=[1.0, 0, 0])
                self.vis.add_geometry(ls)

        if gt_pose is not None:
            self.gt_traj.append(gt_pose)
            frustum = better_camera_frustum(
                gt_pose,
                image.height,
                image.width,
                scale=0.5,
                color=[0, 1.0, 0],
            )
            self.vis.add_geometry(frustum)
            if len(self.gt_traj) > 1:
                ls = poses2lineset(np.stack(self.gt_traj), color=[0, 1.0, 0])
                self.vis.add_geometry(ls)

        if self.n_base_objects > 0:
            if base_objects_color is not None:
                for oid in range(self.n_base_objects):
                    c = base_objects_color[oid]
                    self.base_pcds_vis[oid].paint_uniform_color(c)
                    self.base_bboxes_vis[oid].color = c
            for geom in self.base_pcds_vis + self.base_bboxes_vis:
                self.vis.add_geometry(geom)

        if pcds is not None:
            for i, pcd in enumerate(pcds):
                pcd.transform(est_pose)
                if pcd_colors is not None:
                    pcd.paint_uniform_color(pcd_colors[i][:3])
                self.vis.add_geometry(pcd)

        if new_objects is not None:
            for obj in new_objects:
                pcd = copy.deepcopy(obj["pcd"])
                bbox = copy.deepcopy(obj["bbox"])
                bbox.color = [0.0, 0.0, 1.0]
                if paint_new_objects:
                    pcd.paint_uniform_color([0.0, 1.0, 0.0])
                    bbox.color = [0.0, 1.0, 0.0]
                self.vis.add_geometry(pcd)
                self.vis.add_geometry(bbox)

        self.vis_ctrl.convert_from_pinhole_camera_parameters(self.view_param)
        self.vis.poll_events()
        self.vis.update_renderer()

        rendered = np.asarray(self.vis.capture_screen_float_buffer(False))
        return (rendered, self.vis) if return_vis_handle else (rendered, None)


# ---------------------------------------------------------------------------
# Standalone visualisation helpers
# ---------------------------------------------------------------------------


def get_random_colors(num_colors: int) -> np.ndarray:
    """Generate *num_colors* random RGB colours in [0, 1]."""
    return np.random.rand(num_colors, 3)


def show_mask(
    mask: np.ndarray,
    ax: plt.Axes,
    random_color: bool = False,
) -> None:
    """Overlay a semi-transparent mask on a matplotlib axes."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])])
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    ax.imshow(mask.reshape(h, w, 1) * color.reshape(1, 1, -1))


def show_points(
    coords: np.ndarray,
    labels: np.ndarray,
    ax: plt.Axes,
    marker_size: int = 375,
) -> None:
    """Plot positive/negative keypoints on a matplotlib axes."""
    pos = coords[labels == 1]
    neg = coords[labels == 0]
    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg[:, 0],
        neg[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(
    box: np.ndarray,
    ax: plt.Axes,
    label: str | None = None,
) -> None:
    """Draw a bounding box rectangle on a matplotlib axes."""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle(
            (x0, y0),
            w,
            h,
            edgecolor="green",
            facecolor=(0, 0, 0, 0),
            lw=2,
        )
    )
    if label is not None:
        ax.text(x0, y0, label)


def vis_result_fast(
    image: np.ndarray,
    detections: sv.Detections,
    classes: list[str],
    color: Color | ColorPalette = ColorPalette.DEFAULT,
    instance_random_color: bool = False,
    draw_bbox: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Annotate an image with detection masks/boxes (fast path)."""
    box_ann = sv.BoxAnnotator(color=color, thickness=2)
    label_ann = sv.LabelAnnotator(
        color=color,
        text_scale=0.3,
        text_thickness=1,
        text_padding=2,
    )
    mask_ann = sv.MaskAnnotator(color=color)

    labels: list[str] = []
    if hasattr(detections, "confidence") and hasattr(detections, "class_id"):
        confs = detections.confidence
        cids = detections.class_id
        if confs is not None:
            labels = [f"{classes[c]} {conf:0.2f}" for conf, c in zip(confs, cids)]
        else:
            labels = [f"{classes[c]}" for c in cids]

    if instance_random_color:
        detections = dataclasses.replace(detections)
        detections.class_id = np.arange(len(detections))

    annotated = mask_ann.annotate(scene=image.copy(), detections=detections)
    if draw_bbox:
        annotated = box_ann.annotate(scene=annotated, detections=detections)
        annotated = label_ann.annotate(
            scene=annotated, detections=detections, labels=labels
        )
    return annotated, labels


def vis_result_slow_caption(
    image: np.ndarray,
    masks: np.ndarray,
    boxes_filt: np.ndarray,
    pred_phrases: list[str],
    caption: str,
    text_prompt: str,
) -> np.ndarray:
    """High-quality annotated image (slow, matplotlib-based)."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box, plt.gca(), label)

    plt.title(f"Tagging-Caption: {caption}\nTagging-classes: {text_prompt}\n")
    plt.axis("off")

    fig = plt.gcf()
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return vis_image


def vis_sam_mask(anns: list[dict]) -> np.ndarray:
    """Composite SAM annotation masks into an RGBA image."""
    sorted_anns = sorted(anns, key=lambda x: x["area"], reverse=True)
    h, w = sorted_anns[0]["segmentation"].shape[:2]
    img = np.ones((h, w, 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        img[m] = np.concatenate([np.random.random(3), [0.35]])
    return img


# ---------------------------------------------------------------------------
# Open3D geometry builders
# ---------------------------------------------------------------------------


def poses2lineset(
    poses: np.ndarray, color: list[float] | None = None
) -> o3d.geometry.LineSet:
    """Create a line set connecting sequential camera positions.

    Args:
        poses: (N, 4, 4) camera poses.
        color: RGB colour for the lines (default blue).
    """
    if color is None:
        color = [0, 0, 1]
    n = poses.shape[0]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(poses[:, :3, 3])
    ls.lines = o3d.utility.Vector2iVector(np.array([[i, i + 1] for i in range(n - 1)]))
    ls.colors = o3d.utility.Vector3dVector([color for _ in range(len(ls.lines))])
    return ls


def create_camera_frustum(
    camera_pose: np.ndarray,
    width: float = 1,
    height: float = 1,
    z_near: float = 0.5,
    z_far: float = 1,
    color: list[float] | None = None,
) -> o3d.geometry.LineSet:
    """Simple 5-point camera frustum line set."""
    if color is None:
        color = [0, 0, 1]
    K = np.array(
        [
            [z_near, 0, 0],
            [0, z_near, 0],
            [0, 0, z_near + z_far],
        ]
    )
    points = np.array(
        [
            [-width / 2, -height / 2, z_near],
            [width / 2, -height / 2, z_near],
            [width / 2, height / 2, z_near],
            [-width / 2, height / 2, z_near],
            [0, 0, 0],
        ]
    )
    pts = (camera_pose[:3, :3] @ (K @ points.T) + camera_pose[:3, 3:4]).T

    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(pts)
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 0],
        [4, 1],
        [4, 2],
        [4, 3],
    ]
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    return frustum


def better_camera_frustum(
    camera_pose: np.ndarray | torch.Tensor,
    img_h: int,
    img_w: int,
    scale: float = 3.0,
    color: list[float] | None = None,
) -> o3d.geometry.LineSet:
    """8-point camera frustum with near/far planes."""
    if color is None:
        color = [0, 0, 1]
    if isinstance(camera_pose, torch.Tensor):
        camera_pose = camera_pose.numpy()

    near = scale * 0.1
    far = scale * 1.0
    frustum_h = near
    frustum_w = frustum_h * img_w / img_h

    points = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                d = near if z == -1 else far
                u = x * (frustum_w // 2 if z == -1 else frustum_w * far / near)
                v = y * (frustum_h // 2 if z == -1 else frustum_h * far / near)
                pt = np.array([u, v, d, 1]).reshape(-1, 1)
                points.append((camera_pose @ pt).ravel()[:3])

    lines = [
        [0, 1],
        [1, 3],
        [3, 2],
        [2, 0],
        [4, 5],
        [5, 7],
        [7, 6],
        [6, 4],
        [0, 4],
        [1, 5],
        [3, 7],
        [2, 6],
    ]

    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(points)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    return frustum


# ---------------------------------------------------------------------------
# LineMesh (cylinder-based line rendering)
# ---------------------------------------------------------------------------


_DEFAULT_A = np.array([0, 0, 1])
_DEFAULT_B = np.array([1, 0, 0])


def _align_vector_to_another(
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float | None]:
    """Axis-angle rotation aligning vector *a* to *b*."""
    if a is None:
        a = _DEFAULT_A
    if b is None:
        b = _DEFAULT_B
    if np.array_equal(a, b):
        return None, None
    axis = np.cross(a, b)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.dot(a, b))
    return axis, angle


def _normalized(
    a: np.ndarray, axis: int = -1, order: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Normalise an array of vectors, returning norms as well."""
    norms = np.atleast_1d(np.linalg.norm(a, order, axis))
    norms[norms == 0] = 1
    return a / np.expand_dims(norms, axis), norms


class LineMesh:
    """Sequence of cylinder meshes representing a polyline."""

    def __init__(
        self,
        points: np.ndarray,
        lines: list[list[int]] | np.ndarray | None = None,
        colors: list[float] | np.ndarray | None = None,
        radius: float = 0.15,
    ) -> None:
        if colors is None:
            colors = [0, 1, 0]
        self.points = np.array(points)
        self.lines = (
            np.array(lines)
            if lines is not None
            else self._lines_from_ordered_points(self.points)
        )
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments: list[o3d.geometry.TriangleMesh] = []
        self._create_line_mesh()

    @staticmethod
    def _lines_from_ordered_points(
        points: np.ndarray,
    ) -> np.ndarray:
        return np.array([[i, i + 1] for i in range(points.shape[0] - 1)])

    def _create_line_mesh(self) -> None:
        first = self.points[self.lines[:, 0], :]
        second = self.points[self.lines[:, 1], :]
        segments = second - first
        seg_unit, seg_len = _normalized(segments)
        z_axis = np.array([0, 0, 1])

        for i in range(seg_unit.shape[0]):
            axis, angle = _align_vector_to_another(z_axis, seg_unit[i])
            translation = first[i] + seg_unit[i] * seg_len[i] * 0.5
            cyl = o3d.geometry.TriangleMesh.create_cylinder(self.radius, seg_len[i])
            cyl = cyl.translate(translation, relative=False)
            if axis is not None:
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
                cyl = cyl.rotate(R)
            color = self.colors if self.colors.ndim == 1 else self.colors[i]
            cyl.paint_uniform_color(color)
            self.cylinder_segments.append(cyl)

    def add_line(self, vis: o3d.visualization.Visualizer) -> None:
        """Add all cylinder segments to the visualiser."""
        for cyl in self.cylinder_segments:
            vis.add_geometry(cyl, reset_bounding_box=False)

    def remove_line(self, vis: o3d.visualization.Visualizer) -> None:
        """Remove all cylinder segments from the visualiser."""
        for cyl in self.cylinder_segments:
            vis.remove_geometry(cyl, reset_bounding_box=False)


def save_video_detections(
    exp_out_path: Path,
    save_path: Path | None = None,
    fps: int = 30,
) -> None:
    """Write detection visualisation frames to an MP4 video."""
    if save_path is None:
        save_path = exp_out_path / "vis_video.mp4"

    image_files = sorted((exp_out_path / "vis").glob("*.jpg"))
    first = Image.open(image_files[0])
    width, height = first.size

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
    for f in image_files:
        writer.write(cv2.imread(str(f)))
    writer.release()
    print(f"Video saved at {save_path}")

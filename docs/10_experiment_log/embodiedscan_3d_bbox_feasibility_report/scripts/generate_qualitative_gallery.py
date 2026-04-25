from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from benchmarks.embodiedscan_bbox_feasibility.conceptgraph import (
    generate_conceptgraph_proposals,
)
from benchmarks.embodiedscan_bbox_feasibility.models import EmbodiedScanTarget
from benchmarks.embodiedscan_bbox_feasibility.targets import load_targets
from benchmarks.embodiedscan_eval import (
    _BOX_EDGES,
    compute_oriented_iou_3d,
    oriented_bbox_to_corners,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
REPORT_DIR = REPO_ROOT / "docs/10_experiment_log/embodiedscan_3d_bbox_feasibility_report"
DATA_ROOT = REPO_ROOT / "data/embodiedscan"
SCENE_ROOT = DATA_ROOT / "scannet"
BATCH_DIR = REPO_ROOT / "outputs/embodiedscan_bbox_feasibility/vdetr_batch30_visible_topscenes_fixed"
TARGETS_TSV = BATCH_DIR / "targets.tsv"
DETECTOR_INPUTS = BATCH_DIR / "inputs/detector_inputs.jsonl"
DETECTOR_RECORDS = BATCH_DIR / "detector_outputs/detector_records.jsonl"
SCORES_JSONL = BATCH_DIR / "eval/scores.jsonl"

QUAL_DIR = REPORT_DIR / "resources/qualitative"
DATA_DIR = REPORT_DIR / "resources/data"
GALLERY_MD = REPORT_DIR / "QUALITATIVE_GALLERY.md"
GALLERY_HTML = REPORT_DIR / "qualitative_gallery.html"
LARK_APPEND_XML = REPORT_DIR / "qualitative_lark_append.xml"

CONDITIONS = [
    "single_frame_recon",
    "multi_frame_recon",
    "scannet_pose_crop",
    "scannet_full",
]

METHOD_LABELS = {
    "single_frame_recon": "single",
    "multi_frame_recon": "5-frame",
    "scannet_pose_crop": "crop",
    "scannet_full": "full",
}

COLORS = {
    "gt": "#1a9850",
    "vdetr": "#d73027",
    "cg": "#2c7fb8",
}


@dataclass(frozen=True)
class BatchTarget:
    scene_id: str
    target_id: int
    category: str
    visible_frames: int


@dataclass(frozen=True)
class MethodResult:
    iou: float
    best_index: int | None
    bbox: list[float] | None
    proposal_count: int
    pointcloud_path: Path | None
    frame_ids: list[int]
    num_points: int | None


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    scene_id: str
    target_id: int
    category: str
    visible_frames: int
    rgb_frame_id: int | None
    rgb_path: str | None
    figure_path: Path
    cg_iou: float
    cg_bbox: list[float] | None
    methods: dict[str, MethodResult]


def main() -> None:
    QUAL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    batch_targets = read_batch_targets(TARGETS_TSV)
    targets = load_target_index()
    detector_inputs = read_detector_inputs(DETECTOR_INPUTS)
    detector_records = read_detector_records(DETECTOR_RECORDS)
    scores = read_scores(SCORES_JSONL)

    cg_cache: dict[str, list[dict[str, Any]]] = {}
    cases: list[CaseResult] = []
    for idx, batch_target in enumerate(batch_targets, start=1):
        key = (batch_target.scene_id, batch_target.target_id)
        target = targets[key]
        methods = build_method_results(
            batch_target=batch_target,
            detector_inputs=detector_inputs,
            detector_records=detector_records,
            scores=scores,
        )
        cg_bbox, cg_iou = best_conceptgraph_bbox(target, cg_cache)
        rgb_frame_id, rgb_path = choose_rgb_frame(target)
        figure_path = QUAL_DIR / (
            f"case_{idx:02d}_{safe_slug(batch_target.category)}_"
            f"{batch_target.scene_id}_target{batch_target.target_id}.png"
        )
        case_id = f"Q{idx:02d}"
        case = CaseResult(
            case_id=case_id,
            scene_id=batch_target.scene_id,
            target_id=batch_target.target_id,
            category=batch_target.category,
            visible_frames=batch_target.visible_frames,
            rgb_frame_id=rgb_frame_id,
            rgb_path=str(rgb_path) if rgb_path is not None else None,
            figure_path=figure_path,
            cg_iou=cg_iou,
            cg_bbox=cg_bbox,
            methods=methods,
        )
        render_case_figure(case, target)
        cases.append(case)
        print(f"wrote {figure_path.relative_to(REPORT_DIR)}")

    contact_sheet = render_contact_sheet(cases)
    write_case_csv(cases)
    write_manifest(cases, contact_sheet)
    write_gallery_markdown(cases, contact_sheet)
    write_gallery_html(cases, contact_sheet)
    write_lark_append_xml(cases, contact_sheet)
    print(f"wrote {GALLERY_MD.relative_to(REPO_ROOT)}")
    print(f"wrote {GALLERY_HTML.relative_to(REPO_ROOT)}")
    print(f"wrote {LARK_APPEND_XML.relative_to(REPO_ROOT)}")


def read_batch_targets(path: Path) -> list[BatchTarget]:
    rows: list[BatchTarget] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            scene_id, target_id, category, visible_frames = line.rstrip("\n").split("\t")
            rows.append(
                BatchTarget(
                    scene_id=scene_id,
                    target_id=int(target_id),
                    category=category,
                    visible_frames=int(visible_frames),
                )
            )
    return rows


def load_target_index() -> dict[tuple[str, int], EmbodiedScanTarget]:
    targets = load_targets(
        str(DATA_ROOT),
        split="val",
        source_filter="scannet",
        max_samples=None,
        mini=False,
    )
    return {(target.scene_id, target.target_id): target for target in targets}


def read_detector_inputs(
    path: Path,
) -> dict[tuple[str, int, str], dict[str, Any]]:
    rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            target_id = item.get("target_id")
            if target_id is None:
                continue
            rows[(item["scene_id"], int(target_id), item["input_condition"])] = item
    return rows


def read_detector_records(
    path: Path,
) -> dict[tuple[str, int, str], dict[str, Any]]:
    rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            target_id = item.get("target_id")
            if target_id is None:
                continue
            rows[(item["scene_id"], int(target_id), item["input_condition"])] = item
    return rows


def read_scores(path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            rows[(item["scene_id"], int(item["target_id"]), item["input_condition"])] = item
    return rows


def build_method_results(
    *,
    batch_target: BatchTarget,
    detector_inputs: dict[tuple[str, int, str], dict[str, Any]],
    detector_records: dict[tuple[str, int, str], dict[str, Any]],
    scores: dict[tuple[str, int, str], dict[str, Any]],
) -> dict[str, MethodResult]:
    results: dict[str, MethodResult] = {}
    for condition in CONDITIONS:
        key = (batch_target.scene_id, batch_target.target_id, condition)
        record = detector_records[key]
        score = scores[key]
        input_record = detector_inputs.get(key)
        proposals = record.get("proposals", [])
        best_index = score.get("best_proposal_index")
        bbox = None
        if best_index is not None:
            bbox = proposals[int(best_index)]["bbox_3d"]
        pointcloud_path: Path | None = None
        frame_ids: list[int] = []
        num_points: int | None = None
        if input_record is not None:
            if input_record.get("pointcloud_path"):
                pointcloud_path = REPO_ROOT / input_record["pointcloud_path"]
            frame_ids = [int(v) for v in input_record.get("frame_ids", [])]
            if "num_points" in input_record.get("metadata", {}):
                num_points = int(input_record["metadata"]["num_points"])
        elif record.get("metadata", {}).get("detector_input_path"):
            pointcloud_path = REPO_ROOT / record["metadata"]["detector_input_path"]
        results[condition] = MethodResult(
            iou=float(score["best_iou"]),
            best_index=None if best_index is None else int(best_index),
            bbox=bbox,
            proposal_count=len(proposals),
            pointcloud_path=pointcloud_path,
            frame_ids=frame_ids,
            num_points=num_points,
        )
    return results


def best_conceptgraph_bbox(
    target: EmbodiedScanTarget,
    cache: dict[str, list[dict[str, Any]]],
) -> tuple[list[float] | None, float]:
    if target.scene_id not in cache:
        record = generate_conceptgraph_proposals(
            scene_path=SCENE_ROOT / target.scene_id / "conceptgraph",
            scan_id=target.scan_id,
            scene_id=target.scene_id,
            axis_align_matrix=target.axis_align_matrix,
        )
        cache[target.scene_id] = [proposal.model_dump() for proposal in record.proposals]

    best_bbox: list[float] | None = None
    best_iou = 0.0
    for proposal in cache[target.scene_id]:
        bbox = proposal["bbox_3d"]
        iou = compute_oriented_iou_3d(bbox, target.gt_bbox_3d)
        if iou > best_iou:
            best_iou = float(iou)
            best_bbox = bbox
    return best_bbox, best_iou


def choose_rgb_frame(target: EmbodiedScanTarget) -> tuple[int | None, Path | None]:
    raw_dir = SCENE_ROOT / target.scene_id / "raw"
    candidates = list(dict.fromkeys(target.visible_frame_ids))
    if not candidates:
        candidates = sorted(
            int(path.stem.split("-")[0])
            for path in raw_dir.glob("*-rgb.jpg")
            if path.stem.split("-")[0].isdigit()
        )
    if not candidates:
        return None, None

    image_size = probe_image_size(raw_dir, candidates)
    if image_size is None:
        return None, None
    width, height = image_size

    best_frame: int | None = None
    best_score = -math.inf
    for frame_id in candidates:
        rgb_path = raw_dir / f"{frame_id:06d}-rgb.jpg"
        pose_path = raw_dir / f"{frame_id:06d}.txt"
        intrinsic_path = raw_dir / "intrinsic_color.txt"
        if not rgb_path.exists() or not pose_path.exists() or not intrinsic_path.exists():
            continue
        projected = project_bbox_to_image(
            target.gt_bbox_3d,
            target.axis_align_matrix,
            pose_path,
            intrinsic_path,
        )
        if projected is None:
            continue
        uv, z = projected
        positive = z > 1e-5
        inside = (
            positive
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < height)
        )
        visible_count = int(inside.sum())
        uv_pos = uv[positive]
        clipped_area = 0.0
        if len(uv_pos):
            x0 = max(0.0, float(np.nanmin(uv_pos[:, 0])))
            y0 = max(0.0, float(np.nanmin(uv_pos[:, 1])))
            x1 = min(float(width), float(np.nanmax(uv_pos[:, 0])))
            y1 = min(float(height), float(np.nanmax(uv_pos[:, 1])))
            clipped_area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        center = uv.mean(axis=0)
        center_inside = 0 <= center[0] < width and 0 <= center[1] < height
        score = visible_count * 10000.0 + clipped_area + (5000.0 if center_inside else 0.0)
        if score > best_score:
            best_score = score
            best_frame = frame_id

    if best_frame is None:
        for frame_id in candidates:
            rgb_path = raw_dir / f"{frame_id:06d}-rgb.jpg"
            if rgb_path.exists():
                return frame_id, rgb_path
        return None, None
    return best_frame, raw_dir / f"{best_frame:06d}-rgb.jpg"


def probe_image_size(raw_dir: Path, frame_ids: list[int]) -> tuple[int, int] | None:
    for frame_id in frame_ids:
        path = raw_dir / f"{frame_id:06d}-rgb.jpg"
        if path.exists():
            with Image.open(path) as image:
                return image.size
    return None


def render_case_figure(case: CaseResult, target: EmbodiedScanTarget) -> None:
    fig = plt.figure(figsize=(18, 10.5), dpi=150)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.0], height_ratios=[1, 1])
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1], projection="3d"),
        fig.add_subplot(gs[0, 2], projection="3d"),
        fig.add_subplot(gs[1, 0], projection="3d"),
        fig.add_subplot(gs[1, 1], projection="3d"),
        fig.add_subplot(gs[1, 2]),
    ]
    render_rgb_panel(axes[0], case, target)
    render_pointcloud_panel(
        axes[1],
        case.methods["single_frame_recon"],
        target.gt_bbox_3d,
        title="single RGB-D point cloud",
    )
    render_pointcloud_panel(
        axes[2],
        case.methods["multi_frame_recon"],
        target.gt_bbox_3d,
        title="5-frame RGB-D point cloud",
    )
    render_pointcloud_panel(
        axes[3],
        case.methods["scannet_pose_crop"],
        target.gt_bbox_3d,
        title="raw ScanNet pose crop",
    )
    render_pointcloud_panel(
        axes[4],
        case.methods["scannet_full"],
        target.gt_bbox_3d,
        title="full raw ScanNet scene",
        cg_bbox=case.cg_bbox,
        cg_iou=case.cg_iou,
    )
    render_table_panel(axes[5], case)

    fig.suptitle(
        f"{case.case_id}  {case.scene_id} target {case.target_id} "
        f"({case.category}), visible frames={case.visible_frames}",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.01, 1, 0.96))
    fig.savefig(case.figure_path, bbox_inches="tight")
    plt.close(fig)


def render_rgb_panel(ax: Any, case: CaseResult, target: EmbodiedScanTarget) -> None:
    ax.set_title(
        f"RGB observation frame {case.rgb_frame_id if case.rgb_frame_id is not None else 'N/A'}\n"
        "GT green, V-DETR full red, 2D-CG blue",
        fontsize=10,
    )
    ax.axis("off")
    if case.rgb_path is None:
        ax.text(0.5, 0.5, "RGB frame unavailable", ha="center", va="center")
        return
    image = Image.open(case.rgb_path).convert("RGB")
    ax.imshow(image)
    raw_dir = SCENE_ROOT / case.scene_id / "raw"
    pose_path = raw_dir / f"{case.rgb_frame_id:06d}.txt"
    intrinsic_path = raw_dir / "intrinsic_color.txt"
    draw_projected_bbox(
        ax,
        target.gt_bbox_3d,
        target.axis_align_matrix,
        pose_path,
        intrinsic_path,
        COLORS["gt"],
        "GT",
    )
    full_bbox = case.methods["scannet_full"].bbox
    if full_bbox is not None:
        draw_projected_bbox(
            ax,
            full_bbox,
            target.axis_align_matrix,
            pose_path,
            intrinsic_path,
            COLORS["vdetr"],
            "V-DETR full",
        )
    if case.cg_bbox is not None:
        draw_projected_bbox(
            ax,
            case.cg_bbox,
            target.axis_align_matrix,
            pose_path,
            intrinsic_path,
            COLORS["cg"],
            "2D-CG",
        )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="lower left", fontsize=8, framealpha=0.85)


def render_pointcloud_panel(
    ax: Any,
    result: MethodResult,
    gt_bbox: list[float],
    *,
    title: str,
    cg_bbox: list[float] | None = None,
    cg_iou: float | None = None,
) -> None:
    ax.set_title(f"{title}\nIoU={result.iou:.3f}", fontsize=10)
    ax.set_axis_off()
    if result.pointcloud_path is None or not result.pointcloud_path.exists():
        ax.text2D(0.5, 0.5, "point cloud unavailable", ha="center", va="center")
        return

    points = read_ascii_xyz_ply(result.pointcloud_path)
    focus_boxes = [gt_bbox]
    if result.bbox is not None:
        focus_boxes.append(result.bbox)
    if cg_bbox is not None:
        focus_boxes.append(cg_bbox)
    points_view = crop_points_for_focus(points, focus_boxes)
    points_view = sample_points(points_view, max_points=6500)

    if len(points_view):
        z = points_view[:, 2]
        ax.scatter(
            points_view[:, 0],
            points_view[:, 1],
            points_view[:, 2],
            c=z,
            cmap="viridis",
            s=0.18,
            alpha=0.45,
            linewidths=0,
        )
    plot_bbox_3d(ax, gt_bbox, COLORS["gt"], linewidth=2.0, label="GT")
    if result.bbox is not None:
        plot_bbox_3d(ax, result.bbox, COLORS["vdetr"], linewidth=1.6, label="V-DETR")
    if cg_bbox is not None:
        label = f"2D-CG {cg_iou:.3f}" if cg_iou is not None else "2D-CG"
        plot_bbox_3d(ax, cg_bbox, COLORS["cg"], linewidth=1.6, label=label)
    set_equal_3d_limits(ax, points_view, focus_boxes)
    ax.view_init(elev=24, azim=-58)


def render_table_panel(ax: Any, case: CaseResult) -> None:
    ax.axis("off")
    ax.set_title("Best-box IoU summary", fontsize=11, fontweight="bold")
    rows = [
        [
            "2D-CG",
            f"{case.cg_iou:.3f}",
            "blue",
            "-",
            "-",
        ]
    ]
    for condition in CONDITIONS:
        result = case.methods[condition]
        rows.append(
            [
                METHOD_LABELS[condition],
                f"{result.iou:.3f}",
                "red",
                str(result.proposal_count),
                frame_summary(result.frame_ids),
            ]
        )
    table = ax.table(
        cellText=rows,
        colLabels=["route", "IoU", "pred", "props", "frames"],
        colLoc="left",
        cellLoc="left",
        bbox=[0.0, 0.14, 1.0, 0.75],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1.0, 1.22)
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold")
        if row_idx > 0 and col_idx == 1:
            value = float(cell.get_text().get_text())
            cell.set_facecolor(iou_color(value))
    ax.text(
        0.0,
        0.06,
        "Point-cloud panels plot GT in green. V-DETR best proposals are red. "
        "The full-scene panel also overlays the best 2D-CG proposal in blue.",
        ha="left",
        va="bottom",
        fontsize=8.5,
        wrap=True,
    )


def draw_projected_bbox(
    ax: Any,
    bbox: list[float],
    axis_align_matrix: list[list[float]] | None,
    pose_path: Path,
    intrinsic_path: Path,
    color: str,
    label: str,
) -> None:
    projected = project_bbox_to_image(bbox, axis_align_matrix, pose_path, intrinsic_path)
    if projected is None:
        return
    uv, z = projected
    used_label = False
    for start, end in _BOX_EDGES:
        if z[start] <= 1e-5 or z[end] <= 1e-5:
            continue
        ax.plot(
            [uv[start, 0], uv[end, 0]],
            [uv[start, 1], uv[end, 1]],
            color=color,
            linewidth=2.5,
            label=label if not used_label else None,
        )
        used_label = True


def project_bbox_to_image(
    bbox: list[float],
    axis_align_matrix: list[list[float]] | None,
    pose_path: Path,
    intrinsic_path: Path,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not pose_path.exists() or not intrinsic_path.exists():
        return None
    corners = oriented_bbox_to_corners(bbox)
    if axis_align_matrix is not None:
        axis_align = np.asarray(axis_align_matrix, dtype=np.float64)
        corners = transform_homogeneous(corners, np.linalg.inv(axis_align))
    pose = np.loadtxt(pose_path, dtype=np.float64).reshape(4, 4)
    intrinsic = np.loadtxt(intrinsic_path, dtype=np.float64)[:3, :3]
    corners_camera = transform_homogeneous(corners, np.linalg.inv(pose))
    z = corners_camera[:, 2]
    uvw = (intrinsic @ corners_camera.T).T
    uv = np.zeros((len(corners_camera), 2), dtype=np.float64)
    valid = np.abs(z) > 1e-8
    uv[valid] = uvw[valid, :2] / z[valid, None]
    return uv, z


def transform_homogeneous(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    ones = np.ones((len(points), 1), dtype=np.float64)
    return (matrix @ np.hstack([points[:, :3], ones]).T).T[:, :3]


def read_ascii_xyz_ply(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        vertex_count: int | None = None
        for line in handle:
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY file has no vertex count: {path}")
        points = np.loadtxt(handle, dtype=np.float32, max_rows=vertex_count, usecols=(0, 1, 2))
    if points.ndim == 1:
        points = points.reshape(1, 3)
    if points.shape[1] != 3:
        raise ValueError(f"Expected XYZ point cloud in {path}")
    return points


def crop_points_for_focus(points: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
    centers = np.asarray([box[:3] for box in boxes], dtype=np.float64)
    extents = np.asarray([box[3:6] for box in boxes], dtype=np.float64)
    center = centers[0]
    radius = max(2.5, float(np.nanmax(extents)) * 3.0)
    lo = np.nanmin(centers - extents / 2.0, axis=0) - radius * 0.35
    hi = np.nanmax(centers + extents / 2.0, axis=0) + radius * 0.35
    lo = np.minimum(lo, center - radius)
    hi = np.maximum(hi, center + radius)
    mask = np.all((points >= lo) & (points <= hi), axis=1)
    cropped = points[mask]
    if len(cropped) < 150:
        return points
    return cropped


def sample_points(points: np.ndarray, *, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, num=max_points, dtype=np.int64)
    return points[indices]


def plot_bbox_3d(
    ax: Any,
    bbox: list[float],
    color: str,
    *,
    linewidth: float,
    label: str,
) -> None:
    corners = oriented_bbox_to_corners(bbox)
    used_label = False
    for start, end in _BOX_EDGES:
        ax.plot(
            [corners[start, 0], corners[end, 0]],
            [corners[start, 1], corners[end, 1]],
            [corners[start, 2], corners[end, 2]],
            color=color,
            linewidth=linewidth,
            label=label if not used_label else None,
        )
        used_label = True


def set_equal_3d_limits(
    ax: Any,
    points: np.ndarray,
    boxes: list[list[float]],
) -> None:
    box_corners = np.vstack([oriented_bbox_to_corners(box) for box in boxes])
    if len(points):
        data = np.vstack([points[:, :3], box_corners])
    else:
        data = box_corners
    mins = np.nanmin(data, axis=0)
    maxs = np.nanmax(data, axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.nanmax(maxs - mins)) / 2.0, 0.5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def frame_summary(frame_ids: list[int]) -> str:
    if not frame_ids:
        return "scene"
    if len(frame_ids) <= 3:
        return ",".join(str(v) for v in frame_ids)
    return f"{frame_ids[0]}..{frame_ids[-1]} ({len(frame_ids)})"


def iou_color(value: float) -> str:
    if value >= 0.50:
        return "#d9f0d3"
    if value >= 0.25:
        return "#fff1a8"
    if value > 0:
        return "#fde0dd"
    return "#f5f5f5"


def render_contact_sheet(cases: list[CaseResult]) -> Path:
    thumbs: list[Image.Image] = []
    for case in cases:
        with Image.open(case.figure_path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((520, 310))
            canvas = Image.new("RGB", (540, 350), "white")
            x = (540 - thumb.width) // 2
            canvas.paste(thumb, (x, 10))
            thumbs.append(canvas)

    columns = 3
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * 540, rows * 350), "white")
    for idx, thumb in enumerate(thumbs):
        x = (idx % columns) * 540
        y = (idx // columns) * 350
        sheet.paste(thumb, (x, y))
    out = QUAL_DIR / "qualitative_contact_sheet.png"
    sheet.save(out)
    return out


def write_case_csv(cases: list[CaseResult]) -> None:
    out = DATA_DIR / "qualitative_cases.csv"
    fieldnames = [
        "case_id",
        "scene_id",
        "target_id",
        "category",
        "visible_frames",
        "rgb_frame_id",
        "figure",
        "cg_iou",
        "single_iou",
        "multi_iou",
        "pose_crop_iou",
        "full_scene_iou",
    ]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case_id": case.case_id,
                    "scene_id": case.scene_id,
                    "target_id": case.target_id,
                    "category": case.category,
                    "visible_frames": case.visible_frames,
                    "rgb_frame_id": case.rgb_frame_id,
                    "figure": str(case.figure_path.relative_to(REPORT_DIR)),
                    "cg_iou": f"{case.cg_iou:.6f}",
                    "single_iou": f"{case.methods['single_frame_recon'].iou:.6f}",
                    "multi_iou": f"{case.methods['multi_frame_recon'].iou:.6f}",
                    "pose_crop_iou": f"{case.methods['scannet_pose_crop'].iou:.6f}",
                    "full_scene_iou": f"{case.methods['scannet_full'].iou:.6f}",
                }
            )


def write_manifest(cases: list[CaseResult], contact_sheet: Path) -> None:
    out = DATA_DIR / "qualitative_case_manifest.json"
    payload = {
        "contact_sheet": str(contact_sheet.relative_to(REPORT_DIR)),
        "num_cases": len(cases),
        "cases": [
            {
                "case_id": case.case_id,
                "scene_id": case.scene_id,
                "target_id": case.target_id,
                "category": case.category,
                "visible_frames": case.visible_frames,
                "rgb_frame_id": case.rgb_frame_id,
                "rgb_path": case.rgb_path,
                "figure": str(case.figure_path.relative_to(REPORT_DIR)),
                "ious": {
                    "2d_cg": case.cg_iou,
                    **{condition: result.iou for condition, result in case.methods.items()},
                },
            }
            for case in cases
        ],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_gallery_markdown(cases: list[CaseResult], contact_sheet: Path) -> None:
    lines = [
        "# Qualitative Gallery: RGB, Point Clouds, GT and Best Predictions",
        "",
        "This gallery visualizes every target in the V-DETR batch30 run. Each case",
        "contains one RGB observation panel and four point-cloud panels. Green boxes",
        "are GT, red boxes are the best V-DETR proposal for that input condition,",
        "and blue boxes are the best 2D-CG proposal over the same scene.",
        "",
        f"![Qualitative contact sheet]({contact_sheet.relative_to(REPORT_DIR)})",
        "",
        "## Case Index",
        "",
        "| Case | Scene | Target | Category | 2D-CG | Single | Multi | Pose crop | Full scene | Figure |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for case in cases:
        rel = case.figure_path.relative_to(REPORT_DIR)
        lines.append(
            f"| {case.case_id} | {case.scene_id} | {case.target_id} | {case.category} | "
            f"{case.cg_iou:.3f} | {case.methods['single_frame_recon'].iou:.3f} | "
            f"{case.methods['multi_frame_recon'].iou:.3f} | "
            f"{case.methods['scannet_pose_crop'].iou:.3f} | "
            f"{case.methods['scannet_full'].iou:.3f} | [{rel.name}]({rel}) |"
        )
    lines.extend(["", "## Case Figures", ""])
    for case in cases:
        rel = case.figure_path.relative_to(REPORT_DIR)
        lines.extend(
            [
                f"### {case.case_id}: {case.scene_id} target {case.target_id} ({case.category})",
                "",
                f"- RGB frame: `{case.rgb_frame_id}`",
                f"- IoU summary: 2D-CG `{case.cg_iou:.3f}`, single `{case.methods['single_frame_recon'].iou:.3f}`, "
                f"multi `{case.methods['multi_frame_recon'].iou:.3f}`, pose crop `{case.methods['scannet_pose_crop'].iou:.3f}`, "
                f"full scene `{case.methods['scannet_full'].iou:.3f}`",
                "",
                f"![{case.case_id} qualitative visualization]({rel})",
                "",
            ]
        )
    GALLERY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gallery_html(cases: list[CaseResult], contact_sheet: Path) -> None:
    cards = [
        f'<section class="card wide"><h2>Contact sheet</h2><img src="{contact_sheet.relative_to(REPORT_DIR)}" alt="Qualitative contact sheet"></section>'
    ]
    for case in cases:
        rel = case.figure_path.relative_to(REPORT_DIR)
        cards.append(
            '<section class="card">'
            f"<h2>{case.case_id}: {case.scene_id} target {case.target_id} ({case.category})</h2>"
            f'<img src="{rel}" alt="{case.case_id} qualitative visualization">'
            f"<p>IoU: 2D-CG {case.cg_iou:.3f}, single {case.methods['single_frame_recon'].iou:.3f}, "
            f"multi {case.methods['multi_frame_recon'].iou:.3f}, pose crop {case.methods['scannet_pose_crop'].iou:.3f}, "
            f"full scene {case.methods['scannet_full'].iou:.3f}</p>"
            "</section>"
        )
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>EmbodiedScan Qualitative Gallery</title>
<style>
body{{font-family:Inter,Arial,sans-serif;margin:28px;background:#f7f7f5;color:#202020}}
h1{{margin-bottom:4px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:20px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px;box-shadow:0 1px 2px #0001}}
.card.wide{{grid-column:1/-1}}.card img{{width:100%;height:auto}}code{{background:#eee;padding:2px 4px;border-radius:4px}}
</style>
</head>
<body>
<h1>EmbodiedScan Qualitative Gallery</h1>
<p>Green = GT, red = V-DETR best proposal, blue = 2D-CG best proposal. Generated from local batch30 outputs.</p>
<div class="grid">
{''.join(cards)}
</div>
</body>
</html>
"""
    GALLERY_HTML.write_text(html, encoding="utf-8")


def write_lark_append_xml(cases: list[CaseResult], contact_sheet: Path) -> None:
    lines = [
        "<h1>11. 定性可视化样例：RGB / 点云 / GT / Best Prediction</h1>",
        "<p>下面新增 batch30 全量定性样例。每个 case 图中，绿色框是 GT，红色框是该输入条件下 V-DETR 的 best proposal，蓝色框是同一场景中 2D-CG 的 best proposal。</p>",
        "<table><thead><tr><th background-color=\"light-gray\">Case</th><th background-color=\"light-gray\">Scene</th><th background-color=\"light-gray\">Target</th><th background-color=\"light-gray\">Category</th><th background-color=\"light-gray\">2D-CG</th><th background-color=\"light-gray\">Single</th><th background-color=\"light-gray\">Multi</th><th background-color=\"light-gray\">Pose crop</th><th background-color=\"light-gray\">Full scene</th></tr></thead><tbody>",
    ]
    for case in cases:
        lines.append(
            f"<tr><td>{case.case_id}</td><td>{case.scene_id}</td><td>{case.target_id}</td><td>{case.category}</td>"
            f"<td>{case.cg_iou:.3f}</td><td>{case.methods['single_frame_recon'].iou:.3f}</td>"
            f"<td>{case.methods['multi_frame_recon'].iou:.3f}</td><td>{case.methods['scannet_pose_crop'].iou:.3f}</td>"
            f"<td>{case.methods['scannet_full'].iou:.3f}</td></tr>"
        )
    lines.extend(
        [
            "</tbody></table>",
            f"<p><b>图 Q0：Qualitative contact sheet</b>（本地文件：{contact_sheet.relative_to(REPORT_DIR)}）</p>",
        ]
    )
    for case in cases:
        lines.append(
            f"<p><b>图 {case.case_id}</b>：{case.scene_id} target {case.target_id} "
            f"({case.category})。2D-CG={case.cg_iou:.3f}，single={case.methods['single_frame_recon'].iou:.3f}，"
            f"multi={case.methods['multi_frame_recon'].iou:.3f}，pose crop={case.methods['scannet_pose_crop'].iou:.3f}，"
            f"full scene={case.methods['scannet_full'].iou:.3f}。</p>"
        )
    LARK_APPEND_XML.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


if __name__ == "__main__":
    main()

"""3D object mapping pipeline using Grounded SAM detections.

Models Grounded SAM detections in 3D, assuming tag2text classes and
per-object CLIP features are available on disk.
"""

from __future__ import annotations

import copy
import gzip
import pickle
from pathlib import Path
from typing import Any

import hydra
import imageio
import numpy as np
import omegaconf
import open3d as o3d
import torch
from omegaconf import DictConfig
from PIL import Image
from tqdm import trange

from conceptgraph.dataset.loader import get_dataset
from conceptgraph.slam.mapping import (
    aggregate_similarities,
    compute_spatial_similarities,
    compute_visual_similarities,
    merge_detections_to_objects,
)
from conceptgraph.slam.models import MapObjectList
from conceptgraph.slam.utils import (
    create_or_load_colors,
    denoise_objects,
    filter_objects,
    gobs_to_detection_list,
    merge_obj2_into_obj1,
    merge_objects,
)
from conceptgraph.utils.ious import compute_2d_box_contained_batch
from conceptgraph.utils.vis import OnlineObjectRenderer

BG_CLASSES = ["wall", "floor", "ceiling"]

torch.set_grad_enabled(False)


# -- helpers ---------------------------------------------------------


def _sanitize_path_value(value: Any, dataset_root: Path) -> Any:
    """Recursively replace absolute paths with dataset-root-relative strings."""
    if isinstance(value, dict):
        return {k: _sanitize_path_value(v, dataset_root) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_path_value(v, dataset_root) for v in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_path_value(v, dataset_root) for v in value)

    if isinstance(value, Path):
        value = str(value)

    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            resolved = candidate.resolve()
            try:
                return resolved.relative_to(dataset_root).as_posix()
            except ValueError:
                rel = Path(
                    str(resolved.resolve()).replace(str(dataset_root), "").lstrip("/")
                )
                return rel.as_posix()

    return value


def _sanitize_cfg_for_export(cfg: DictConfig) -> DictConfig:
    """Return a copy of *cfg* with absolute paths made relative."""
    dataset_root = Path(cfg.dataset_root).resolve()
    cfg_container = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    sanitized = _sanitize_path_value(cfg_container, dataset_root)
    return omegaconf.OmegaConf.create(sanitized)


def compute_match_batch(
    cfg: DictConfig,
    spatial_sim: torch.Tensor,
    visual_sim: torch.Tensor,
) -> torch.Tensor:
    """Compute detection-to-object assignment from similarity matrices.

    Args:
        cfg: Pipeline config with ``match_method``, ``phys_bias``, and
            ``sim_threshold``.
        spatial_sim: ``(M, N)`` spatial similarity tensor.
        visual_sim: ``(M, N)`` visual similarity tensor.

    Returns:
        ``(M, N)`` binary assignment matrix. Each row has at most one 1,
        meaning a detection maps to at most one existing object.
    """
    assign_mat = torch.zeros_like(spatial_sim)
    if cfg.match_method == "sim_sum":
        sims = (1 + cfg.phys_bias) * spatial_sim + (1 - cfg.phys_bias) * visual_sim
        row_max, row_argmax = torch.max(sims, dim=1)
        for i in row_max.argsort(descending=True):
            if row_max[i] > cfg.sim_threshold:
                assign_mat[i, row_argmax[i]] = 1
            else:
                break
    else:
        raise ValueError(f"Unknown matching method: {cfg.match_method}")

    return assign_mat


def prepare_objects_save_vis(
    objects: MapObjectList,
    downsample_size: float = 0.025,
) -> list[dict]:
    """Downsample and strip objects to only the keys needed for saving."""
    objects_to_save = copy.deepcopy(objects)

    keep_keys = {
        "pcd",
        "bbox",
        "clip_ft",
        "text_ft",
        "class_id",
        "num_detections",
        "inst_color",
    }

    for i in range(len(objects_to_save)):
        objects_to_save[i]["pcd"] = objects_to_save[i]["pcd"].voxel_down_sample(
            downsample_size
        )

    for i in range(len(objects_to_save)):
        for k in list(objects_to_save[i].keys()):
            if k not in keep_keys:
                del objects_to_save[i][k]

    return objects_to_save.to_serializable()


def process_cfg(cfg: DictConfig) -> DictConfig:
    """Resolve dataset paths and image dimensions from config."""
    cfg.dataset_root = Path(cfg.dataset_root)
    cfg.dataset_config = Path(cfg.dataset_config)

    if cfg.dataset_config.name != "multiscan.yaml":
        dataset_cfg = omegaconf.OmegaConf.load(cfg.dataset_config)
        if cfg.image_height is None:
            cfg.image_height = dataset_cfg.camera_params.image_height
        if cfg.image_width is None:
            cfg.image_width = dataset_cfg.camera_params.image_width
        print(
            f"Setting image height and width to "
            f"{cfg.image_height} x {cfg.image_width}"
        )
    else:
        assert (
            cfg.image_height is not None and cfg.image_width is not None
        ), "For multiscan dataset, image height and width must be specified"

    return cfg


# -- main loop -------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="../configs/slam_pipeline",
    config_name="base",
)
def main(cfg: DictConfig) -> None:
    cfg = process_cfg(cfg)

    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device="cpu",
        dtype=torch.float,
    )

    classes, class_colors = create_or_load_colors(cfg, cfg.color_file_name)

    objects = MapObjectList(device=cfg.device)

    if not cfg.skip_bg:
        bg_objects: dict[str, Any] | None = dict.fromkeys(BG_CLASSES)
    else:
        bg_objects = None

    if cfg.vis_render:
        view_param = o3d.io.read_pinhole_camera_parameters(cfg.render_camera_path)
        obj_renderer = OnlineObjectRenderer(
            view_param=view_param,
            base_objects=None,
            gray_map=False,
        )
        frames: list[np.ndarray] = []

    if cfg.save_objects_all_frames:
        save_all_folder = (
            cfg.dataset_root
            / cfg.scene_id
            / "objects_all_frames"
            / f"{cfg.gsa_variant}_{cfg.save_suffix}"
        )
        save_all_folder.mkdir(parents=True, exist_ok=True)

    for idx in trange(len(dataset)):
        color_path = Path(dataset.color_paths[idx])
        image_original_pil = Image.open(color_path)

        color_tensor, depth_tensor, intrinsics, *_ = dataset[idx]

        color_np = color_tensor.cpu().numpy()
        image_rgb = color_np.astype(np.uint8)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"

        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()

        cam_K = intrinsics.cpu().numpy()[:3, :3]

        detections_path = (
            cfg.dataset_root
            / cfg.scene_id
            / cfg.detection_folder_name
            / color_path.name
        ).with_suffix(".pkl.gz")

        with gzip.open(detections_path, "rb") as f:
            gobs = pickle.load(f)

        unt_pose = dataset.poses[idx].cpu().numpy()
        adjusted_pose = unt_pose

        fg_detection_list, bg_detection_list = gobs_to_detection_list(
            cfg=cfg,
            image=image_rgb,
            depth_array=depth_array,
            cam_K=cam_K,
            idx=idx,
            gobs=gobs,
            trans_pose=adjusted_pose,
            class_names=classes,
            bg_classes=BG_CLASSES,
            color_path=str(color_path),
        )

        if len(bg_detection_list) > 0 and bg_objects is not None:
            for detected_object in bg_detection_list:
                class_name = detected_object["class_name"][0]
                if bg_objects[class_name] is None:
                    bg_objects[class_name] = detected_object
                else:
                    bg_objects[class_name] = merge_obj2_into_obj1(
                        cfg,
                        bg_objects[class_name],
                        detected_object,
                        run_dbscan=False,
                    )

        if len(fg_detection_list) == 0:
            continue

        if cfg.use_contain_number:
            xyxy = fg_detection_list.get_stacked_values_torch("xyxy", 0)
            contain_numbers = compute_2d_box_contained_batch(
                xyxy, cfg.contain_area_thresh
            )
            for i in range(len(fg_detection_list)):
                fg_detection_list[i]["contain_number"] = [contain_numbers[i]]

        if len(objects) == 0:
            for i in range(len(fg_detection_list)):
                objects.append(fg_detection_list[i])
            continue

        spatial_sim = compute_spatial_similarities(cfg, fg_detection_list, objects)
        visual_sim = compute_visual_similarities(cfg, fg_detection_list, objects)
        agg_sim = aggregate_similarities(cfg, spatial_sim, visual_sim)

        if cfg.use_contain_number:
            contain_numbers_objects = torch.Tensor(
                [obj["contain_number"][0] for obj in objects]
            )
            detection_contained = (contain_numbers > 0).unsqueeze(1)
            object_contained = (contain_numbers_objects > 0).unsqueeze(0)
            xor = detection_contained ^ object_contained
            agg_sim[xor] -= cfg.contain_mismatch_penalty

        agg_sim[agg_sim < cfg.sim_threshold] = float("-inf")

        objects = merge_detections_to_objects(cfg, fg_detection_list, objects, agg_sim)

        if cfg.denoise_interval > 0 and (idx + 1) % cfg.denoise_interval == 0:
            objects = denoise_objects(cfg, objects)
        if cfg.filter_interval > 0 and (idx + 1) % cfg.filter_interval == 0:
            objects = filter_objects(cfg, objects)
        if cfg.merge_interval > 0 and (idx + 1) % cfg.merge_interval == 0:
            objects = merge_objects(cfg, objects)

        if cfg.save_objects_all_frames:
            save_all_path = save_all_folder / f"{idx:06d}.pkl.gz"
            objects_to_save = MapObjectList(
                [o for o in objects if o["num_detections"] >= cfg.obj_min_detections]
            )
            objects_to_save = prepare_objects_save_vis(objects_to_save)

            if not cfg.skip_bg:
                bg_objects_to_save = MapObjectList(
                    [v for v in bg_objects.values() if v is not None]
                )
                bg_objects_to_save = prepare_objects_save_vis(bg_objects_to_save)
            else:
                bg_objects_to_save = None

            result = {
                "camera_pose": adjusted_pose,
                "objects": objects_to_save,
                "bg_objects": bg_objects_to_save,
            }
            with gzip.open(save_all_path, "wb") as f:
                pickle.dump(result, f)

        if cfg.vis_render:
            objects_vis = MapObjectList(
                [
                    copy.deepcopy(o)
                    for o in objects
                    if o["num_detections"] >= cfg.obj_min_detections
                ]
            )

            if cfg.class_agnostic:
                objects_vis.color_by_instance()
            else:
                objects_vis.color_by_most_common_classes(class_colors)

            rendered_image, vis = obj_renderer.step(
                image=image_original_pil,
                gt_pose=adjusted_pose,
                new_objects=objects_vis,
                paint_new_objects=False,
                return_vis_handle=cfg.debug_render,
            )

            if cfg.debug_render:
                vis.run()
                del vis

            if rendered_image is not None:
                rendered_image = (rendered_image * 255).astype(np.uint8)
                frames.append(rendered_image)

    # -- post-processing ------------------------------------------------

    if bg_objects is not None:
        bg_objects = MapObjectList([v for v in bg_objects.values() if v is not None])
        bg_objects = denoise_objects(cfg, bg_objects)

    objects = denoise_objects(cfg, objects)

    if cfg.save_pcd:
        results = {
            "objects": objects.to_serializable(),
            "bg_objects": (
                None if bg_objects is None else bg_objects.to_serializable()
            ),
            "cfg": _sanitize_cfg_for_export(cfg),
            "class_names": classes,
            "class_colors": class_colors,
        }

        pcd_save_path = (
            cfg.dataset_root
            / cfg.scene_id
            / "pcd_saves"
            / f"full_pcd_{cfg.gsa_variant}_{cfg.save_suffix}.pkl.gz"
        )
        pcd_save_path.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(pcd_save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved full point cloud to {pcd_save_path}")

    objects = filter_objects(cfg, objects)
    objects = merge_objects(cfg, objects)

    if cfg.save_pcd:
        results["objects"] = objects.to_serializable()
        pcd_post_path = pcd_save_path.with_name(
            pcd_save_path.stem.replace(".pkl", "") + "_post.pkl.gz"
        )
        with gzip.open(pcd_post_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved full point cloud after post-processing to " f"{pcd_post_path}")

    if cfg.save_objects_all_frames:
        save_meta_path = save_all_folder / "meta.pkl.gz"
        with gzip.open(save_meta_path, "wb") as f:
            pickle.dump(
                {
                    "cfg": _sanitize_cfg_for_export(cfg),
                    "class_names": classes,
                    "class_colors": class_colors,
                },
                f,
            )

    if cfg.vis_render:
        objects_vis = MapObjectList(
            [o for o in objects if o["num_detections"] >= cfg.obj_min_detections]
        )

        if cfg.class_agnostic:
            objects_vis.color_by_instance()
        else:
            objects_vis.color_by_most_common_classes(class_colors)

        rendered_image, vis = obj_renderer.step(
            image=image_original_pil,
            gt_pose=adjusted_pose,
            new_objects=objects_vis,
            paint_new_objects=False,
            return_vis_handle=False,
        )

        rendered_image = (rendered_image * 255).astype(np.uint8)
        frames.append(rendered_image)

        stacked_frames = np.stack(frames)
        video_save_path = (
            cfg.dataset_root
            / cfg.scene_id
            / f"objects_mapping-{cfg.gsa_variant}-{cfg.save_suffix}.mp4"
        )
        imageio.mimwrite(str(video_save_path), stacked_frames, fps=10)
        print(f"Save video to {video_save_path}")


if __name__ == "__main__":
    main()

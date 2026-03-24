from __future__ import annotations

import copy
from collections.abc import Iterable

import matplotlib
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from conceptgraph.utils.general import to_numpy, to_tensor


class DetectionList(list):
    """Typed list of detection dicts with convenience accessors."""

    def get_values(self, key: str, idx: int | None = None) -> list:
        """Retrieve *key* from every detection."""
        if idx is None:
            return [d[key] for d in self]
        return [d[key][idx] for d in self]

    def get_stacked_values_torch(
        self, key: str, idx: int | None = None
    ) -> torch.Tensor:
        """Stack values for *key* into a single tensor."""
        values = []
        for d in self:
            v = d[key]
            if idx is not None:
                v = v[idx]
            if isinstance(
                v,
                (
                    o3d.geometry.OrientedBoundingBox,
                    o3d.geometry.AxisAlignedBoundingBox,
                ),
            ):
                v = np.asarray(v.get_box_points())
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            values.append(v)
        return torch.stack(values, dim=0)

    def get_stacked_values_numpy(self, key: str, idx: int | None = None) -> np.ndarray:
        """Stack values for *key* into a numpy array."""
        return to_numpy(self.get_stacked_values_torch(key, idx))

    def __add__(self, other: DetectionList) -> DetectionList:
        new_list = copy.deepcopy(self)
        new_list.extend(other)
        return new_list

    def __iadd__(self, other: DetectionList) -> DetectionList:
        self.extend(other)
        return self

    def slice_by_indices(self, index: Iterable[int]) -> DetectionList:
        """Return a sub-list by integer indices."""
        new = type(self)()
        for i in index:
            new.append(self[i])
        return new

    def slice_by_mask(self, mask: Iterable[bool]) -> DetectionList:
        """Return a sub-list filtered by a boolean mask."""
        new = type(self)()
        for item, keep in zip(self, mask):
            if keep:
                new.append(item)
        return new

    def get_most_common_class(self) -> list[int]:
        """Most frequent class id per detection."""
        classes = []
        for d in self:
            vals, counts = np.unique(np.asarray(d["class_id"]), return_counts=True)
            classes.append(vals[np.argmax(counts)])
        return classes

    def color_by_most_common_classes(
        self,
        colors_dict: dict[str, list[float]],
        color_bbox: bool = True,
    ) -> None:
        """Paint each detection's point cloud by its dominant class."""
        classes = self.get_most_common_class()
        for d, c in zip(self, classes):
            color = colors_dict[str(c)]
            d["pcd"].paint_uniform_color(color)
            if color_bbox:
                d["bbox"].color = color

    def color_by_instance(self) -> None:
        """Assign a unique colour per detection instance."""
        if len(self) == 0:
            return
        if "inst_color" in self[0]:
            for d in self:
                d["pcd"].paint_uniform_color(d["inst_color"])
                d["bbox"].color = d["inst_color"]
        else:
            cmap = matplotlib.colormaps.get_cmap("turbo")
            colors = cmap(np.linspace(0, 1, len(self)))[:, :3]
            for i, d in enumerate(self):
                d["pcd"].paint_uniform_color(colors[i])
                d["bbox"].color = colors[i]


class MapObjectList(DetectionList):
    """Persistent map objects with CLIP features and serialisation."""

    def compute_similarities(
        self, new_clip_ft: np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """Cosine similarity of *new_clip_ft* against all objects."""
        new_clip_ft = to_tensor(new_clip_ft)
        clip_fts = self.get_stacked_values_torch("clip_ft")
        return F.cosine_similarity(new_clip_ft.unsqueeze(0), clip_fts)

    def to_serializable(self) -> list[dict]:
        """Convert to a list of plain dicts (numpy arrays only)."""
        out: list[dict] = []
        for obj in self:
            d = copy.deepcopy(obj)
            d["clip_ft"] = to_numpy(d["clip_ft"])
            d["text_ft"] = to_numpy(d["text_ft"])
            d["pcd_np"] = np.asarray(d["pcd"].points)
            d["bbox_np"] = np.asarray(d["bbox"].get_box_points())
            d["pcd_color_np"] = np.asarray(d["pcd"].colors)
            del d["pcd"], d["bbox"]
            out.append(d)
        return out

    def load_serializable(self, s_obj_list: list[dict]) -> None:
        """Populate from a serialised list (inverse of *to_serializable*)."""
        assert len(self) == 0, "Must be empty before loading"
        for s in s_obj_list:
            obj = copy.deepcopy(s)
            obj["clip_ft"] = to_tensor(obj["clip_ft"])
            obj["text_ft"] = to_tensor(obj["text_ft"])

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj["pcd_np"])
            pcd.colors = o3d.utility.Vector3dVector(obj["pcd_color_np"])

            bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(obj["bbox_np"])
            )
            bbox.color = obj["pcd_color_np"][0]

            obj["pcd"] = pcd
            obj["bbox"] = bbox
            del obj["pcd_np"], obj["bbox_np"], obj["pcd_color_np"]
            self.append(obj)

"""
BEV Builder: Extensible Bird's Eye View rendering for different datasets.

This module provides a base class and dataset-specific implementations for
generating annotated BEV images suitable for multimodal LLM input.

Architecture:
    BaseBEVBuilder (abstract)
    ├── ReplicaBEVBuilder   - For Replica dataset scenes
    ├── ScanNetBEVBuilder   - For ScanNet dataset (future)
    └── CustomBEVBuilder    - For custom scenes with centroid data

Usage:
    # Using factory
    builder = create_bev_builder("replica", config=BEVConfig(image_size=1000))
    img, path, labels = builder.build(scene_objects)

    # Direct instantiation
    builder = ReplicaBEVBuilder()
    img, path, labels = builder.build(objects)
"""

from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d


@dataclass
class BEVConfig:
    """Configuration for BEV generation."""

    image_size: int = 1500  # Higher resolution for better quality
    padding: float = 0.08  # meters of padding around scene
    object_diameter: int = 20  # circle diameter in pixels
    font_scale: float = 0.45
    font_thickness: int = 1
    label_offset: tuple[int, int] = (14, 5)
    background_color: tuple[int, int, int] = (248, 248, 248)
    object_color: tuple[int, int, int] = (50, 50, 210)  # Red-ish for visibility
    text_color: tuple[int, int, int] = (25, 25, 25)
    border_color: tuple[int, int, int] = (180, 180, 180)
    show_legend: bool = False  # Disable legend when mesh is shown
    show_title: bool = False  # Disable title when mesh is shown
    title: str = "Scene BEV (annotated)"
    max_labels_per_quadrant: int = 25  # Limit labels to avoid overcrowding
    category_colors: dict[str, tuple[int, int, int]] | None = None
    # Coordinate system options
    flip_y: bool = False  # Flip Y axis (for different coordinate conventions)
    swap_xy: bool = False  # Swap X and Y axes
    # Mesh rendering options
    mesh_path: str | Path | None = None  # Path to mesh PLY file
    render_mesh: bool = True  # Whether to render mesh as background
    show_objects: bool = True  # Whether to draw object markers
    show_labels: bool = True  # Whether to draw object labels
    # Camera view options
    perspective: bool = True  # True: perspective from above, False: orthographic
    camera_fov: float = (
        100.0  # Field of view in degrees (larger = stronger perspective)
    )


@dataclass
class AnnotatedObject:
    """Object annotation for BEV visualization."""

    obj_id: int
    category: str
    centroid_3d: tuple[float, float, float]  # (x, y, z) in world space
    centroid_2d: tuple[float, float] = (0.0, 0.0)  # (x, y) projected for BEV
    pixel_pos: tuple[int, int] = (0, 0)  # (x, y) in pixel space
    label: str = ""  # Display label (e.g., "001: sofa")
    metadata: dict[str, Any] = field(default_factory=dict)  # Extra data


class BaseBEVBuilder(ABC):
    """
    Abstract base class for BEV image generation.

    Subclasses implement dataset-specific object extraction logic.
    The rendering pipeline is shared across all implementations.
    """

    def __init__(self, config: BEVConfig | None = None):
        self.config = config or BEVConfig()

    @abstractmethod
    def extract_annotations(self, objects: list[Any]) -> list[AnnotatedObject]:
        """
        Extract annotations from dataset-specific object format.

        This method must be implemented by subclasses to handle
        the specific data format of each dataset.

        Args:
            objects: List of scene objects in dataset-specific format

        Returns:
            List of AnnotatedObject with centroid_3d and category
        """
        pass

    def build(
        self,
        objects: list[Any],
        output_path: str | Path | None = None,
        mesh_path: str | Path | None = None,
    ) -> tuple[np.ndarray, Path, dict[int, str]]:
        """
        Build annotated BEV image from scene objects.

        Args:
            objects: List of scene objects (format depends on dataset)
            output_path: Optional path to save image (creates temp if None)
            mesh_path: Optional path to mesh PLY file for background rendering

        Returns:
            Tuple of (image_array, image_path, obj_id_to_label_map)
        """
        if not objects:
            return self._create_empty_image(output_path)

        # Extract dataset-specific annotations
        annotations = self.extract_annotations(objects)
        if not annotations:
            return self._create_empty_image(output_path)

        # Project 3D centroids to 2D BEV (for orthographic mode)
        self._project_to_2d(annotations)

        # Determine mesh path
        effective_mesh_path = mesh_path or self.config.mesh_path

        # Render with mesh if available
        if effective_mesh_path and self.config.render_mesh:
            img, transform = self._render_mesh_open3d(effective_mesh_path)
            # Transform annotations to pixel coordinates
            for ann in annotations:
                # Use 3D centroid for perspective projection
                px, py = self._world_to_pixel_open3d(
                    np.array(ann.centroid_3d), transform, self.config.image_size
                )
                ann.pixel_pos = (px, py)
        else:
            # No mesh: create blank image and use object-based transform
            centroids = np.array([a.centroid_2d for a in annotations])
            transform = self._compute_transform(centroids)
            for ann in annotations:
                px, py = self._world_to_pixel(np.array(ann.centroid_2d), transform)
                ann.pixel_pos = (px, py)
            img = np.ones(
                (self.config.image_size, self.config.image_size, 3), dtype=np.uint8
            )
            img[:] = self.config.background_color

        # Draw objects and labels on top of rendered mesh (if enabled)
        if self.config.show_objects:
            self._draw_objects(img, annotations)
        if self.config.show_labels:
            self._draw_labels(img, annotations)

        # Draw legend if enabled
        if self.config.show_legend:
            self._draw_legend(img, annotations)

        # Add title if enabled
        if self.config.show_title:
            cv2.putText(
                img,
                self.config.title,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.config.text_color,
                1,
            )

        # Build label map
        obj_id_to_label = {ann.obj_id: ann.label for ann in annotations}

        # Save image
        output_path = self._save_image(img, output_path)

        return img, output_path, obj_id_to_label

    def _render_mesh_open3d(
        self, mesh_path: str | Path
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Render mesh as BEV image using OpenCV triangle rasterization.

        Supports two view modes:
        - Orthographic (perspective=False): Pure top-down view
        - Perspective (perspective=True): 3/4 isometric-like view with depth

        If the PLY file has polygon faces that fail to decompose into triangles,
        performs Ball Pivoting surface reconstruction to create a proper mesh.

        Args:
            mesh_path: Path to the PLY file

        Returns:
            Tuple of (rendered_image, transform_info)

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If geometry has no colors
        """
        mesh_path = Path(mesh_path)
        if not mesh_path.exists():
            raise FileNotFoundError(f"File not found: {mesh_path}")

        size = self.config.image_size

        # Try loading as triangle mesh first
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        # Check if we have a valid mesh with enough triangles
        min_expected_triangles = len(vertices) // 10
        if len(triangles) < min_expected_triangles:
            mesh = self._reconstruct_mesh_from_pointcloud(mesh_path)
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)

        if not mesh.has_vertex_colors():
            raise ValueError(f"Mesh has no vertex colors: {mesh_path}")

        colors = np.asarray(mesh.vertex_colors)

        # Filter ceiling triangles
        ceiling_threshold = self._detect_ceiling_threshold(vertices[:, 2])
        face_max_z = vertices[triangles].max(axis=1)[:, 2]
        keep_mask = face_max_z < ceiling_threshold
        kept_triangles = triangles[keep_mask]

        # Choose rendering mode
        if self.config.perspective:
            img, transform = self._render_perspective(
                vertices, colors, kept_triangles, size
            )
        else:
            img, transform = self._render_orthographic(
                vertices, colors, kept_triangles, size
            )

        return img, transform

    def _render_orthographic(
        self,
        vertices: np.ndarray,
        colors: np.ndarray,
        triangles: np.ndarray,
        size: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Render with orthographic (top-down) projection."""
        padding = 0.05
        x_min, y_min = vertices[:, :2].min(axis=0)
        x_max, y_max = vertices[:, :2].max(axis=0)
        max_range = max(x_max - x_min, y_max - y_min)
        scale = size * (1 - 2 * padding) / max_range
        offset_x = size / 2 - (x_max + x_min) / 2 * scale
        offset_y = size / 2 - (y_max + y_min) / 2 * scale

        transform = {
            "mode": "orthographic",
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "scale": scale,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "size": size,
        }

        # Project to 2D (simple XY drop Z)
        tri_verts = vertices[triangles]
        px = (tri_verts[:, :, 0] * scale + offset_x).astype(np.int32)
        py = (size - (tri_verts[:, :, 1] * scale + offset_y)).astype(np.int32)
        pts = np.stack([px, py], axis=-1)

        tri_colors = colors[triangles]
        avg_colors_bgr = (tri_colors.mean(axis=1) * 255).astype(np.uint8)[:, ::-1]

        img = np.ones((size, size, 3), dtype=np.uint8)
        img[:, :] = self.config.background_color

        for j in range(len(pts)):
            cv2.fillPoly(img, [pts[j]], tuple(int(c) for c in avg_colors_bgr[j]))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, transform

    def _render_perspective(
        self,
        vertices: np.ndarray,
        colors: np.ndarray,
        triangles: np.ndarray,
        size: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Render with perspective projection from directly above.

        Camera is positioned at scene center looking straight down,
        with perspective projection that makes edge objects show their sides
        (near-far perspective effect).
        """
        # Scene bounds
        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)
        center = (bounds_min + bounds_max) / 2
        extent = bounds_max - bounds_min

        # Camera directly above scene center
        # Lower height = stronger perspective effect
        max_xy = max(extent[0], extent[1])
        camera_height = max_xy * 0.6  # Relatively close for strong perspective
        camera_pos = np.array([center[0], center[1], bounds_max[2] + camera_height])

        # Camera looks straight down (-Z)
        # We want: world X -> image X (right), world Y -> image -Y (down)
        # So: right = [1, 0, 0], up_cam = [0, 1, 0] (but image Y is flipped)
        right = np.array([1.0, 0.0, 0.0])
        up_cam = np.array([0.0, 1.0, 0.0])
        forward = np.array([0.0, 0.0, -1.0])

        # View matrix: transforms world coords to camera coords
        # Camera X = world X, Camera Y = world Y, Camera Z = -world Z
        R = np.stack([right, -up_cam, forward], axis=0)
        t = -R @ camera_pos

        # Wide FOV for strong perspective effect
        fov = np.radians(self.config.camera_fov)
        f = size / (2 * np.tan(fov / 2))
        cx, cy = size / 2, size / 2

        # Transform all vertices to camera space
        vertices_cam = (R @ vertices.T).T + t
        z_cam = np.clip(vertices_cam[:, 2], 0.01, None)

        # Project to image plane
        x_img = f * vertices_cam[:, 0] / z_cam + cx
        y_img = f * vertices_cam[:, 1] / z_cam + cy

        transform = {
            "mode": "perspective",
            "R": R,
            "t": t,
            "f": f,
            "cx": cx,
            "cy": cy,
            "camera_pos": camera_pos,
            "center": center,
            "size": size,
        }

        # Sort triangles by depth (painter's algorithm)
        tri_verts_cam = vertices_cam[triangles]
        tri_depths = tri_verts_cam[:, :, 2].mean(axis=1)
        depth_order = np.argsort(-tri_depths)

        # Prepare triangle coordinates and colors
        tri_x = x_img[triangles].astype(np.int32)
        tri_y = y_img[triangles].astype(np.int32)
        tri_pts = np.stack([tri_x, tri_y], axis=-1)
        tri_colors = (colors[triangles].mean(axis=1) * 255).astype(np.uint8)[:, ::-1]

        img = np.ones((size, size, 3), dtype=np.uint8)
        img[:, :] = self.config.background_color

        # Render in depth order
        for idx in depth_order:
            pts = tri_pts[idx]
            if np.any(pts < -500) or np.any(pts > size + 500):
                continue
            cv2.fillPoly(img, [pts], tuple(int(c) for c in tri_colors[idx]))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, transform

    def _world_to_pixel_open3d(
        self,
        point_3d: np.ndarray,
        transform: dict[str, Any],
        image_size: int,
    ) -> tuple[int, int]:
        """
        Convert world coordinates to pixel coordinates for rendered image.

        Args:
            point_3d: 3D point (x, y, z) in world coordinates
            transform: Transform info from rendering
            image_size: Image size (unused, kept for compatibility)

        Returns:
            (px, py) pixel coordinates
        """
        mode = transform.get("mode", "orthographic")

        if mode == "orthographic":
            scale = transform["scale"]
            offset_x = transform["offset_x"]
            offset_y = transform["offset_y"]
            size = transform["size"]

            px = int(point_3d[0] * scale + offset_x)
            py = int(size - (point_3d[1] * scale + offset_y))
        else:
            # Perspective projection
            R = transform["R"]
            t = transform["t"]
            f = transform["f"]
            cx = transform["cx"]
            cy = transform["cy"]

            # Use full 3D point for perspective projection
            point_cam = R @ point_3d[:3] + t
            z_cam = max(point_cam[2], 0.01)

            px = int(f * point_cam[0] / z_cam + cx)
            py = int(f * point_cam[1] / z_cam + cy)

        return px, py

    def _project_to_2d(self, annotations: list[AnnotatedObject]) -> None:
        """
        Project 3D centroids to 2D for BEV.

        Default: Use X, Y axes (top-down view from Z+).
        Override in subclass for different projections.
        """
        config = self.config
        for ann in annotations:
            x, y, z = ann.centroid_3d
            if config.swap_xy:
                x, y = y, x
            if config.flip_y:
                y = -y
            ann.centroid_2d = (x, y)

    def _compute_transform(self, centroids: np.ndarray) -> dict[str, Any]:
        """Compute world-to-pixel transformation parameters."""
        padding = self.config.padding
        size = self.config.image_size

        min_pt = centroids.min(axis=0) - padding
        max_pt = centroids.max(axis=0) + padding

        # Compute scale to fit in image
        range_x = max_pt[0] - min_pt[0]
        range_y = max_pt[1] - min_pt[1]
        scale = (size * 0.85) / max(range_x, range_y)

        # Compute offset to center
        offset = np.array(
            [
                (size - range_x * scale) / 2,
                (size - range_y * scale) / 2,
            ]
        )

        return {
            "min_pt": min_pt,
            "max_pt": max_pt,
            "scale": scale,
            "offset": offset,
        }

    def _world_to_pixel(
        self, point: np.ndarray, transform: dict[str, Any]
    ) -> tuple[int, int]:
        """Convert world coordinates to pixel coordinates."""
        min_pt = transform["min_pt"]
        scale = transform["scale"]
        offset = transform["offset"]

        x = int((point[0] - min_pt[0]) * scale + offset[0])
        y = int((point[1] - min_pt[1]) * scale + offset[1])
        return x, y

    def _reconstruct_mesh_from_pointcloud(
        self, mesh_path: str | Path
    ) -> o3d.geometry.TriangleMesh:
        """
        Reconstruct triangle mesh from point cloud using Ball Pivoting.

        This is used when the PLY file has polygon faces that Open3D
        cannot decompose into triangles.

        Results are cached to disk for faster subsequent loads.

        Args:
            mesh_path: Path to the PLY file

        Returns:
            Reconstructed triangle mesh with vertex colors
        """
        mesh_path = Path(mesh_path)

        # Check for cached reconstructed mesh
        cache_path = mesh_path.parent / f"{mesh_path.stem}_triangulated.ply"
        if cache_path.exists():
            mesh = o3d.io.read_triangle_mesh(str(cache_path))
            if len(mesh.triangles) > 0 and mesh.has_vertex_colors():
                return mesh

        # Load as point cloud
        pcd = o3d.io.read_point_cloud(str(mesh_path))
        if not pcd.has_colors():
            raise ValueError(f"Point cloud has no colors: {mesh_path}")

        # Ensure normals exist for Ball Pivoting
        if not pcd.has_normals():
            pcd.estimate_normals()

        # Compute average point distance for radius estimation
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)

        # Ball Pivoting surface reconstruction
        # Use multiple radii to handle varying point density
        radii = [avg_dist * 2, avg_dist * 4]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )

        # Transfer colors from point cloud to mesh
        mesh.vertex_colors = pcd.colors

        # Cache the reconstructed mesh for future use
        o3d.io.write_triangle_mesh(str(cache_path), mesh)

        return mesh

    def _detect_ceiling_threshold(self, z: np.ndarray) -> float:
        """Use histogram to detect ceiling Z threshold."""
        hist, bin_edges = np.histogram(z, bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        z_range = z.max() - z.min()
        ceiling_start = z.max() - z_range * 0.25
        ceiling_mask = bin_centers > ceiling_start

        if not np.any(ceiling_mask):
            return z.max()

        ceiling_bins = hist[ceiling_mask]
        ceiling_centers = bin_centers[ceiling_mask]
        peak_idx = np.argmax(ceiling_bins)
        ceiling_peak_z = ceiling_centers[peak_idx]

        # Threshold: 0.15m below the ceiling peak
        return ceiling_peak_z - 0.15

    def _draw_objects(
        self, img: np.ndarray, annotations: list[AnnotatedObject]
    ) -> None:
        """Draw object circles on image with white outline for visibility."""
        config = self.config
        category_colors = config.category_colors or {}

        for ann in annotations:
            x, y = ann.pixel_pos

            # Get color (category-specific or default)
            color = category_colors.get(ann.category, config.object_color)

            # Draw white outline first for visibility on dark backgrounds
            cv2.circle(img, (x, y), config.object_diameter // 2 + 1, (255, 255, 255), 2)
            # Draw filled circle
            cv2.circle(img, (x, y), config.object_diameter // 2 - 1, color, -1)

    def _draw_labels(self, img: np.ndarray, annotations: list[AnnotatedObject]) -> None:
        """Draw labels with quadrant-based collision avoidance."""
        config = self.config
        size = config.image_size

        # Group by quadrant (0=top-right, 1=top-left, 2=bottom-left, 3=bottom-right)
        quadrants: dict[int, list[AnnotatedObject]] = {0: [], 1: [], 2: [], 3: []}

        for ann in annotations:
            x, y = ann.pixel_pos
            q = 0
            if x < size // 2:
                q = 1 if y < size // 2 else 2
            else:
                q = 0 if y < size // 2 else 3
            quadrants[q].append(ann)

        # Draw labels per quadrant (limit to max_labels_per_quadrant)
        for q, anns in quadrants.items():
            # Sort by y position for vertical stacking
            anns.sort(key=lambda a: a.pixel_pos[1])
            anns = anns[: config.max_labels_per_quadrant]

            for ann in anns:
                x, y = ann.pixel_pos
                label = ann.label

                (text_w, text_h), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    config.font_scale,
                    config.font_thickness,
                )

                # Offset direction based on quadrant
                ox, oy = config.label_offset
                if q in (1, 2):  # Left quadrants
                    lx = x - text_w - ox
                else:  # Right quadrants
                    lx = x + ox

                if q in (0, 1):  # Top quadrants
                    ly = y - oy
                else:  # Bottom quadrants
                    ly = y + text_h + oy

                # Clamp to image bounds
                lx = max(2, min(size - text_w - 2, lx))
                ly = max(text_h + 2, min(size - 2, ly))

                # Draw background rectangle
                cv2.rectangle(
                    img,
                    (lx - 1, ly - text_h - 1),
                    (lx + text_w + 1, ly + 2),
                    (255, 255, 255),
                    -1,
                )
                cv2.rectangle(
                    img,
                    (lx - 1, ly - text_h - 1),
                    (lx + text_w + 1, ly + 2),
                    (200, 200, 200),
                    1,
                )

                # Draw text
                cv2.putText(
                    img,
                    label,
                    (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    config.font_scale,
                    config.text_color,
                    config.font_thickness,
                )

    def _draw_legend(self, img: np.ndarray, annotations: list[AnnotatedObject]) -> None:
        """Draw category count legend."""
        config = self.config
        size = config.image_size

        # Count categories
        category_counts: dict[str, int] = {}
        for ann in annotations:
            cat = ann.category
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Sort by count
        sorted_cats = sorted(category_counts.items(), key=lambda x: -x[1])[:10]

        # Draw legend in bottom-right
        legend_x = size - 150
        legend_y = size - 20 - len(sorted_cats) * 15

        cv2.putText(
            img,
            "Categories:",
            (legend_x, legend_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            config.text_color,
            1,
        )

        for i, (cat, count) in enumerate(sorted_cats):
            text = f"{cat[:12]}: {count}"
            cv2.putText(
                img,
                text,
                (legend_x, legend_y + 12 + i * 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (80, 80, 80),
                1,
            )

    def _create_empty_image(
        self, output_path: str | Path | None
    ) -> tuple[np.ndarray, Path, dict[int, str]]:
        """Create empty placeholder image."""
        size = self.config.image_size
        img = np.ones((size, size, 3), dtype=np.uint8)
        img[:] = self.config.background_color

        cv2.putText(
            img,
            "No objects",
            (size // 2 - 50, size // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.config.text_color,
            2,
        )

        output_path = self._save_image(img, output_path, prefix="bev_empty_")
        return img, output_path, {}

    def _save_image(
        self,
        img: np.ndarray,
        output_path: str | Path | None,
        prefix: str = "bev_",
    ) -> Path:
        """Save image to file."""
        if output_path is None:
            fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix=prefix)
            output_path = Path(tmp_path)
        else:
            output_path = Path(output_path)

        cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return output_path


class ReplicaBEVBuilder(BaseBEVBuilder):
    """
    BEV builder for Replica dataset scenes.

    Expects objects in format:
    - Dict with 'pcd_np' (point cloud) and 'class_name' (list of categories)
    - Or SceneObject with .pcd_np, .centroid, .category attributes
    """

    def extract_annotations(self, objects: list[Any]) -> list[AnnotatedObject]:
        """Extract annotations from Replica scene objects."""
        annotations = []

        for i, obj in enumerate(objects):
            # Handle both dict and object-style access
            if isinstance(obj, dict):
                pcd_np = obj.get("pcd_np")
                centroid = obj.get("centroid")
                class_names = obj.get("class_name", [])
                category_attr = obj.get("category", "")
            else:
                pcd_np = getattr(obj, "pcd_np", None)
                centroid = getattr(obj, "centroid", None)
                class_names = getattr(obj, "class_name", [])
                category_attr = getattr(obj, "category", "")

            # Get centroid: prefer explicit centroid, then compute from pcd_np
            if centroid is not None:
                centroid_3d = tuple(float(c) for c in centroid[:3])
            elif pcd_np is not None and len(pcd_np) > 0:
                pcd_np = np.asarray(pcd_np)
                centroid_3d = tuple(float(c) for c in np.mean(pcd_np, axis=0)[:3])
            else:
                continue  # Skip objects without position data

            # Get category: prefer category attribute, then class_name list
            category = ""
            if category_attr:
                category = str(category_attr)
            elif class_names and len(class_names) > 0 and class_names[0]:
                category = str(class_names[0])
            else:
                category = f"obj_{i}"

            # Create label with global numbering
            label = f"{i:03d}: {category}"

            annotations.append(
                AnnotatedObject(
                    obj_id=i,
                    category=category,
                    centroid_3d=centroid_3d,
                    label=label,
                )
            )

        return annotations

    # Backward compatibility alias
    def generate(
        self,
        objects: list[Any],
        output_path: str | Path | None = None,
        mesh_path: str | Path | None = None,
    ) -> tuple[np.ndarray, Path, dict[int, str]]:
        """Alias for build() - for backward compatibility."""
        return self.build(objects, output_path, mesh_path=mesh_path)


class GenericBEVBuilder(BaseBEVBuilder):
    """
    Generic BEV builder for custom scenes.

    Flexible input format supporting both dicts and objects with:
    - centroid / centroid_3d / position / xyz
    - category / class_name / label / type
    """

    def extract_annotations(self, objects: list[Any]) -> list[AnnotatedObject]:
        """Extract annotations from generic scene objects."""
        annotations = []

        for i, obj in enumerate(objects):
            # Try to get centroid from various field names
            centroid_3d = self._get_centroid(obj, i)
            if centroid_3d is None:
                continue

            # Try to get category from various field names
            category = self._get_category(obj, i)

            # Create label
            label = f"{i:03d}: {category}"

            annotations.append(
                AnnotatedObject(
                    obj_id=i,
                    category=category,
                    centroid_3d=centroid_3d,
                    label=label,
                )
            )

        return annotations

    def _get_centroid(self, obj: Any, idx: int) -> tuple[float, float, float] | None:
        """Try to extract centroid from object using various field names."""
        field_names = [
            "centroid",
            "centroid_3d",
            "position",
            "xyz",
            "center",
            "loc",
            "location",
        ]

        for name in field_names:
            val = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
            if val is not None:
                val = np.asarray(val).flatten()
                if len(val) >= 3:
                    return (float(val[0]), float(val[1]), float(val[2]))
                elif len(val) >= 2:
                    return (float(val[0]), float(val[1]), 0.0)

        # Try pcd_np as fallback
        pcd_np = (
            obj.get("pcd_np") if isinstance(obj, dict) else getattr(obj, "pcd_np", None)
        )
        if pcd_np is not None and len(pcd_np) > 0:
            pcd_np = np.asarray(pcd_np)
            c = np.mean(pcd_np, axis=0)
            if len(c) >= 3:
                return (float(c[0]), float(c[1]), float(c[2]))

        return None

    def _get_category(self, obj: Any, idx: int) -> str:
        """Try to extract category from object using various field names."""
        field_names = ["category", "class_name", "label", "type", "name", "tag"]

        for name in field_names:
            val = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
            if val is not None:
                if isinstance(val, list) and len(val) > 0:
                    return str(val[0])
                elif isinstance(val, str) and val:
                    return val

        return f"obj_{idx}"


# Alias for backward compatibility
SceneBEVGenerator = ReplicaBEVBuilder


# ============================================================================
# OpenEQA ScanNet BEV Builder
# ============================================================================


@dataclass
class OpenEQAScanNetBEVConfig:
    """Configuration for OpenEQA ScanNet BEV rendering."""

    image_size: int = 1500
    fov_deg: float = 80.0
    camera_height_factor: float = 0.7  # camera height = max_xy_extent * factor
    frustum_stride: int = 10  # sample every Nth camera for visibility check
    frustum_margin_px: int = 50  # pixel margin for frustum visibility
    frustum_max_depth: float = 10.0  # max depth in meters
    ceiling_normal_threshold: float = -0.5  # faces with normal_z below this are ceiling
    trajectory_color: tuple[int, int, int] = (59, 130, 246)  # blue
    trajectory_width: int = 2
    background_color: tuple[int, int, int] = (245, 245, 245)
    # ScanNet data root for automatic mesh discovery
    scannet_data_root: str = "data/scannetv2"


class OpenEQAScanNetBEVBuilder:
    """BEV builder for OpenEQA ScanNet scenes using full-resolution ScanNet meshes.

    This builder produces high-quality bird's-eye view images by:
    1. Loading the full-resolution ``_vh_clean.ply`` mesh from ScanNet v2
    2. Filtering triangles to only those visible from the camera trajectory (frustum test)
    3. Removing downward-facing ceiling triangles via surface normal filtering
    4. Rendering with perspective projection from above using painter's algorithm
    5. Overlaying the camera trajectory path

    Usage::

        builder = OpenEQAScanNetBEVBuilder()
        img, path = builder.build(
            mesh_path="data/scannetv2/scene0709_00/scene0709_00_vh_clean.ply",
            traj_path="data/OpenEQA/scannet/002-scannet-scene0709_00/conceptgraph/traj.txt",
            intrinsic_path="data/OpenEQA/scannet/002-scannet-scene0709_00/raw/intrinsic_color.txt",
            output_path="bev/scene_bev.png",
        )

    Or auto-discover all paths from a scene path::

        builder = OpenEQAScanNetBEVBuilder()
        img, path = builder.build_from_scene(
            scene_path="data/OpenEQA/scannet/002-scannet-scene0709_00/conceptgraph",
            output_path="bev/scene_bev.png",
        )
    """

    def __init__(self, config: OpenEQAScanNetBEVConfig | None = None) -> None:
        self.config = config or OpenEQAScanNetBEVConfig()

    def build(
        self,
        mesh_path: str | Path,
        traj_path: str | Path,
        intrinsic_path: str | Path,
        output_path: str | Path | None = None,
        img_w: int = 1296,
        img_h: int = 968,
    ) -> tuple[np.ndarray, Path]:
        """Render a BEV image from ScanNet mesh filtered by camera frustum.

        Args:
            mesh_path: Path to ``<scanId>_vh_clean.ply``
            traj_path: Path to ``traj.txt`` (4x4 camera-to-world per line)
            intrinsic_path: Path to ``intrinsic_color.txt`` (4x4 intrinsic matrix)
            output_path: Where to save the PNG (auto-generated if None)
            img_w: Original image width for frustum calculation
            img_h: Original image height for frustum calculation

        Returns:
            Tuple of (rendered_image_array, output_path)
        """
        cfg = self.config
        mesh_path = Path(mesh_path)
        traj_path = Path(traj_path)
        intrinsic_path = Path(intrinsic_path)

        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")
        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectory not found: {traj_path}")
        if not intrinsic_path.exists():
            raise FileNotFoundError(f"Intrinsics not found: {intrinsic_path}")

        # Load mesh
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.compute_triangle_normals()
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        tris = np.asarray(mesh.triangles)
        colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
        normals = np.asarray(mesh.triangle_normals)

        # Load camera data
        K = np.loadtxt(str(intrinsic_path))[:3, :3]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        traj = np.loadtxt(str(traj_path)).reshape(-1, 4, 4)
        cam_positions = traj[:, :3, 3]

        # Step 1: Frustum visibility filtering
        visible = self._compute_frustum_visibility(
            verts, traj, fx, fy, cx, cy, img_w, img_h
        )
        tri_visible = visible[tris[:, 0]] & visible[tris[:, 1]] & visible[tris[:, 2]]

        # Step 2: Remove ceiling faces (facing downward)
        facing_down = normals[:, 2] < cfg.ceiling_normal_threshold
        tri_keep = tri_visible & (~facing_down)
        kept_tris = tris[tri_keep]

        if len(kept_tris) == 0:
            # Fallback: render all visible tris without ceiling filter
            kept_tris = tris[tri_visible]

        # Step 3: Perspective rendering from above
        img = self._render_perspective(verts, colors, kept_tris, cam_positions)

        # Step 4: Overlay camera trajectory
        img = self._draw_trajectory(img, verts, cam_positions)

        # Save
        if output_path is None:
            fd, tmp = tempfile.mkstemp(suffix=".png", prefix="bev_scannet_")
            output_path = Path(tmp)
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return img, output_path

    def _config_hash(self) -> str:
        """Compute a short hash of the current config for cache keying."""
        import hashlib
        from dataclasses import asdict

        d = asdict(self.config)
        s = str(sorted(d.items()))
        return hashlib.md5(s.encode()).hexdigest()[:8]

    def build_from_scene(
        self,
        scene_path: str | Path,
        output_path: str | Path | None = None,
    ) -> tuple[np.ndarray, Path]:
        """Auto-discover mesh, trajectory, and intrinsics from a scene path.

        Searches for the ScanNet mesh in ``data/scannetv2/<scene_id>/`` based on
        the scene_info.json or clip directory naming convention.

        BEV images are cached to ``<scene_path>/bev/scene_bev_scannet_<config_hash>.png``
        inside the conceptgraph folder. Subsequent calls with the same config
        return the cached image without re-rendering.

        Args:
            scene_path: Path to the scene's conceptgraph directory
                        (e.g., ``data/OpenEQA/scannet/002-scannet-scene0709_00/conceptgraph``)
            output_path: Where to save the PNG. If None, auto-generates a
                         cache path under ``scene_path/bev/``.

        Returns:
            Tuple of (rendered_image_array, output_path)
        """
        scene_path = Path(scene_path)
        cfg = self.config
        scannet_root = Path(cfg.scannet_data_root)

        # Resolve conceptgraph dir
        cg_dir = scene_path if scene_path.name == "conceptgraph" else scene_path
        clip_dir = cg_dir.parent if cg_dir.name == "conceptgraph" else cg_dir
        clip_name = clip_dir.name

        # Default output path with config hash for caching
        if output_path is None:
            bev_dir = cg_dir / "bev"
            bev_dir.mkdir(parents=True, exist_ok=True)
            output_path = bev_dir / f"scene_bev_scannet_{self._config_hash()}.png"

        # Check cache
        output_path = Path(output_path)
        if output_path.exists():
            img = cv2.imread(str(output_path))
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), output_path

        # Discover ScanNet scene ID
        scan_id = self._extract_scan_id(clip_name, cg_dir)

        # Find mesh
        mesh_path = scannet_root / scan_id / f"{scan_id}_vh_clean.ply"
        if not mesh_path.exists():
            mesh_path = scannet_root / scan_id / f"{scan_id}_vh_clean_2.ply"
        if not mesh_path.exists():
            raise FileNotFoundError(
                f"No ScanNet mesh found for {scan_id} in {scannet_root}. "
                f"Run: python scripts/download_scannet_mesh.py --scenes {scan_id} --types _vh_clean.ply"
            )

        # Find trajectory
        traj_path = cg_dir / "traj.txt"
        if not traj_path.exists():
            traj_path = clip_dir / "conceptgraph" / "traj.txt"

        # Find intrinsics
        intrinsic_path = clip_dir / "raw" / "intrinsic_color.txt"
        if not intrinsic_path.exists():
            intrinsic_path = cg_dir / "intrinsic_color.txt"

        return self.build(
            mesh_path=mesh_path,
            traj_path=traj_path,
            intrinsic_path=intrinsic_path,
            output_path=output_path,
        )

    def _extract_scan_id(self, clip_name: str, scene_path: Path) -> str:
        """Extract ScanNet scene ID from clip directory name or scene_info.json."""
        # Try scene_info.json first
        info_path = scene_path / "scene_info.json"
        if not info_path.exists():
            info_path = scene_path.parent / "conceptgraph" / "scene_info.json"
        if info_path.exists():
            import json
            info = json.loads(info_path.read_text())
            scene_id = info.get("scene_id")
            if scene_id:
                return scene_id

        # Parse from clip name: "002-scannet-scene0709_00" -> "scene0709_00"
        parts = clip_name.split("-", 2)
        if len(parts) >= 3 and parts[2].startswith("scene"):
            return parts[2]

        # Last resort: use clip name as-is
        return clip_name

    def _compute_frustum_visibility(
        self,
        verts: np.ndarray,
        traj: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        img_w: int,
        img_h: int,
    ) -> np.ndarray:
        """Compute per-vertex visibility across all camera frustums."""
        cfg = self.config
        visible = np.zeros(len(verts), dtype=bool)
        margin = cfg.frustum_margin_px

        for i in range(0, len(traj), cfg.frustum_stride):
            pose = traj[i]
            if np.any(np.isnan(pose)):
                continue
            w2c = np.linalg.inv(pose)
            pts_cam = (w2c[:3, :3] @ verts.T).T + w2c[:3, 3]
            z = pts_cam[:, 2]
            u = fx * pts_cam[:, 0] / np.maximum(z, 0.01) + cx
            v = fy * pts_cam[:, 1] / np.maximum(z, 0.01) + cy
            visible |= (
                (z > 0.1)
                & (u >= -margin)
                & (u < img_w + margin)
                & (v >= -margin)
                & (v < img_h + margin)
                & (z < cfg.frustum_max_depth)
            )
        return visible

    def _render_perspective(
        self,
        verts: np.ndarray,
        colors: np.ndarray,
        tris: np.ndarray,
        cam_positions: np.ndarray,
    ) -> np.ndarray:
        """Render triangles with perspective projection from above.

        Uses a virtual camera positioned above the scene center looking straight
        down, with near-to-far painter's algorithm so floor/furniture surfaces
        overwrite any remaining ceiling fragments.
        """
        cfg = self.config
        size = cfg.image_size

        # Compute scene bounds from visible geometry
        used_vert_ids = np.unique(tris.ravel())
        vis_verts = verts[used_vert_ids]
        bmin = vis_verts.min(axis=0)
        bmax = vis_verts.max(axis=0)
        center = (bmin + bmax) / 2
        extent = bmax - bmin

        # Virtual camera above scene center
        max_xy = max(extent[0], extent[1])
        cam_height = max_xy * cfg.camera_height_factor
        eye = np.array([center[0], center[1], bmax[2] + cam_height])

        # View matrix: X→right, -Y→down, -Z→forward (looking down)
        R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        t = -R @ eye

        # Perspective projection
        fov = np.radians(cfg.fov_deg)
        f = size / (2.0 * np.tan(fov / 2.0))
        c = size / 2.0

        # Store transform for trajectory drawing
        self._view_R = R
        self._view_t = t
        self._view_f = f
        self._view_c = c

        # Project all vertices
        v_cam = (R @ verts.T).T + t
        z_cam = np.clip(v_cam[:, 2], 0.01, None)
        x_img = f * v_cam[:, 0] / z_cam + c
        y_img = f * v_cam[:, 1] / z_cam + c

        # Sort triangles near-to-far (near = small Z in camera = top of scene)
        # This ensures floor/furniture overwrites ceiling remnants
        tri_v_cam = v_cam[tris]
        tri_depths = tri_v_cam[:, :, 2].mean(axis=1)
        order = np.argsort(tri_depths)

        # Prepare triangle pixel coordinates and colors
        tri_x = x_img[tris].astype(np.int32)
        tri_y = y_img[tris].astype(np.int32)
        tri_pts = np.stack([tri_x, tri_y], axis=-1)

        tri_colors = colors[tris]
        avg_bgr = (tri_colors.mean(axis=1) * 255).astype(np.uint8)[:, ::-1]

        # Rasterize
        img = np.ones((size, size, 3), dtype=np.uint8)
        img[:] = cfg.background_color
        for j in order:
            p = tri_pts[j]
            if np.any(p < -500) or np.any(p > size + 500):
                continue
            cv2.fillPoly(img, [p], tuple(int(c_val) for c_val in avg_bgr[j]))

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _draw_trajectory(
        self,
        img: np.ndarray,
        verts: np.ndarray,
        cam_positions: np.ndarray,
    ) -> np.ndarray:
        """Draw camera trajectory on the BEV image."""
        cfg = self.config
        R, t, f, c = self._view_R, self._view_t, self._view_f, self._view_c

        for i in range(len(cam_positions) - 1):
            p1_w, p2_w = cam_positions[i], cam_positions[i + 1]
            if np.any(np.isnan(p1_w)) or np.any(np.isnan(p2_w)):
                continue
            p1_cam = R @ p1_w + t
            p2_cam = R @ p2_w + t
            if p1_cam[2] < 0.01 or p2_cam[2] < 0.01:
                continue
            p1x = f * p1_cam[0] / p1_cam[2] + c
            p1y = f * p1_cam[1] / p1_cam[2] + c
            p2x = f * p2_cam[0] / p2_cam[2] + c
            p2y = f * p2_cam[1] / p2_cam[2] + c
            if any(np.isnan(v) or np.isinf(v) for v in (p1x, p1y, p2x, p2y)):
                continue
            cv2.line(
                img,
                (int(p1x), int(p1y)),
                (int(p2x), int(p2y)),
                cfg.trajectory_color,
                cfg.trajectory_width,
                cv2.LINE_AA,
            )

        return img


OpenEQADefaultBEVConfig = OpenEQAScanNetBEVConfig()


# ============================================================================
# Default Configs
# ============================================================================

# Default BEV config for Replica scenes (used by KeyframeSelector and LLMEvaluator)
# - Perspective view from above
# - Mesh-only rendering (no object markers or labels)
# - 1000x1000 resolution for LLM visual understanding
ReplicaDefaultBEVConfig = BEVConfig(
    image_size=1000,
    perspective=True,
    show_objects=False,
    show_labels=False,
)


def create_bev_builder(
    dataset: str = "replica",
    config: BEVConfig | None = None,
) -> BaseBEVBuilder:
    """
    Factory function to create dataset-specific BEV builder.

    Args:
        dataset: Dataset name ("replica", "scannet", "generic")
        config: Optional BEVConfig

    Returns:
        BaseBEVBuilder instance

    Raises:
        ValueError: If dataset is not supported
    """
    builders = {
        "replica": ReplicaBEVBuilder,
        "generic": GenericBEVBuilder,
        "scannet": ReplicaBEVBuilder,  # ScanNet uses same object format
        "openeqa_scannet": None,  # Use OpenEQAScanNetBEVBuilder directly
    }

    if dataset not in builders:
        supported = ", ".join(builders.keys())
        raise ValueError(f"Unknown dataset '{dataset}'. Supported: {supported}")

    return builders[dataset](config=config)

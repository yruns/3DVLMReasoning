"""Offscreen scene-graph visualization (headless-friendly).

Renders a scene graph to static images, PLY models, or interactive HTML
without requiring a display. Supports multiple output formats:

1. Static images from several preset viewpoints
2. PLY point-cloud / mesh files
3. Interactive HTML via plotly (with matplotlib fallback)
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import distinctipy
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (side-effect import)

from conceptgraph.slam.models import MapObjectList
from conceptgraph.utils.vis import LineMesh

# -- geometry helpers ------------------------------------------------


def _create_ball_mesh(
    center: np.ndarray,
    radius: float,
    color: tuple[float, float, float] = (0, 1, 0),
) -> o3d.geometry.TriangleMesh:
    """Create a coloured sphere mesh at *center*."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh.translate(center)
    mesh.paint_uniform_color(color)
    return mesh


# -- I/O helpers -----------------------------------------------------


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offscreen scene-graph visualization")
    parser.add_argument(
        "--result_path",
        type=str,
        required=True,
        help="Path to the scene-graph map file (.pkl.gz)",
    )
    parser.add_argument(
        "--edge_file",
        type=str,
        default=None,
        help="Path to the object-relation JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./visualization_output",
        help="Output directory",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="images",
        choices=["images", "ply", "html", "all"],
        help="Output format",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=1920,
        help="Output image width",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=1080,
        help="Output image height",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=8,
        help="Number of viewpoints to render",
    )
    parser.add_argument(
        "--original_mesh",
        type=str,
        default=None,
        help="Path to the ground-truth scene mesh (e.g. from Replica)",
    )
    return parser


def _load_result(
    result_path: str | Path,
) -> tuple[MapObjectList, MapObjectList | None, dict]:
    """Load a scene-graph result file.

    Returns:
        Tuple of (objects, bg_objects, class_colors).
    """
    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)

    if isinstance(results, dict):
        objects = MapObjectList()
        objects.load_serializable(results["objects"])

        if results["bg_objects"] is None:
            bg_objects = None
        else:
            bg_objects = MapObjectList()
            bg_objects.load_serializable(results["bg_objects"])

        class_colors = results["class_colors"]

    elif isinstance(results, list):
        objects = MapObjectList()
        objects.load_serializable(results)

        bg_objects = None
        class_colors = distinctipy.get_colors(len(objects), pastel_factor=0.5)
        class_colors = {str(i): c for i, c in enumerate(class_colors)}
    else:
        raise ValueError(f"Unknown results type: {type(results)}")

    return objects, bg_objects, class_colors


# -- scene-graph geometry construction -------------------------------


def _create_scene_graph_geometries(
    objects: MapObjectList,
    edges: list[dict],
    class_colors: dict,
) -> tuple[list, list[np.ndarray]]:
    """Build Open3D geometries (node spheres + edge cylinders).

    Returns:
        Tuple of (geometry_list, object_centers).
    """
    geometries: list = []

    classes = objects.get_most_common_class()
    colors = [class_colors[str(c)] for c in classes]
    obj_centers: list[np.ndarray] = []

    for obj, c in zip(objects, colors):
        points = np.asarray(obj["pcd"].points)
        center = np.mean(points, axis=0)
        obj_centers.append(center)

        ball = _create_ball_mesh(center, radius=0.10, color=c)
        geometries.append(ball)

    for edge in edges:
        if edge["object_relation"] == "none of these":
            continue
        id1 = edge["object1"]["id"]
        id2 = edge["object2"]["id"]

        line_mesh = LineMesh(
            points=np.array([obj_centers[id1], obj_centers[id2]]),
            lines=np.array([[0, 1]]),
            colors=[1, 0, 0],
            radius=0.02,
        )
        geometries.extend(line_mesh.cylinder_segments)

    return geometries, obj_centers


# -- image output ----------------------------------------------------


def _save_as_images_matplotlib(
    objects: MapObjectList,
    scene_graph_geometries: list,
    output_dir: Path,
    num_views: int,
    image_width: int,
    image_height: int,
    original_mesh_path: str | None = None,
) -> None:
    """Generate multi-view images using matplotlib (fallback renderer)."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    original_mesh_points: np.ndarray | None = None
    if original_mesh_path and Path(original_mesh_path).exists():
        print(f"  Loading original mesh: {original_mesh_path}")
        try:
            mesh = o3d.io.read_triangle_mesh(original_mesh_path)
            if len(mesh.vertices) > 0:
                pcd = mesh.sample_points_uniformly(number_of_points=50_000)
                original_mesh_points = np.asarray(pcd.points)
                print(
                    f"  Loaded original mesh " f"({len(original_mesh_points)} points)"
                )
        except Exception as exc:
            print(f"  Failed to load original mesh: {exc}")

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    for obj in objects:
        points = np.asarray(obj["pcd"].points)
        colors = np.asarray(obj["pcd"].colors)

        if len(points) > 500:
            indices = np.random.choice(len(points), 500, replace=False)
            points = points[indices]
            colors = colors[indices]

        all_points.append(points)
        all_colors.append(colors)

    all_pts = np.vstack(all_points)
    all_clrs = np.vstack(all_colors)

    node_positions: list[np.ndarray] = []
    for geom in scene_graph_geometries:
        if isinstance(geom, o3d.geometry.TriangleMesh):
            vertices = np.asarray(geom.vertices)
            if len(vertices) > 0:
                node_positions.append(np.mean(vertices, axis=0))

    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

    for i, angle in enumerate(angles):
        fig = plt.figure(figsize=(image_width / 100, image_height / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")

        if original_mesh_points is not None:
            ax.scatter(
                original_mesh_points[:, 0],
                original_mesh_points[:, 1],
                original_mesh_points[:, 2],
                c="lightgray",
                s=0.5,
                alpha=0.3,
                label="Original Scene",
            )

        ax.scatter(
            all_pts[:, 0],
            all_pts[:, 1],
            all_pts[:, 2],
            c=all_clrs,
            s=1,
            alpha=0.6,
            label="Objects",
        )

        if node_positions:
            nodes = np.array(node_positions)
            ax.scatter(
                nodes[:, 0],
                nodes[:, 1],
                nodes[:, 2],
                c="yellow",
                s=100,
                marker="o",
                edgecolors="black",
                linewidths=2,
            )

        ax.view_init(elev=20, azim=np.degrees(angle))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Scene Graph View {i + 1}/{num_views}")
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")

        image_path = images_dir / f"scene_graph_view_{i:02d}.png"
        plt.savefig(image_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved view {i + 1}/{num_views}: {image_path}")

    print(f"\nAll images saved to: {images_dir}")


def _save_as_images(
    objects: MapObjectList,
    bg_objects: MapObjectList | None,
    scene_graph_geometries: list,
    output_dir: Path,
    image_width: int,
    image_height: int,
    num_views: int,
    original_mesh_path: str | None = None,
) -> None:
    """Render multi-view images via Open3D offscreen, falling back to matplotlib."""
    print(f"\nGenerating {num_views} viewpoint images...")

    pcds = copy.deepcopy(objects.get_values("pcd"))
    bboxes = copy.deepcopy(objects.get_values("bbox"))

    for i in range(len(objects)):
        pcds[i] = pcds[i].voxel_down_sample(0.05)

    try:
        vis = o3d.visualization.Visualizer()
        success = vis.create_window(
            width=image_width, height=image_height, visible=False
        )
        if not success:
            raise RuntimeError("Failed to create Open3D window")

        for geometry in pcds + bboxes + scene_graph_geometries:
            vis.add_geometry(geometry)

        opt = vis.get_render_option()
        if opt is None:
            raise RuntimeError("Failed to get render options")
        opt.background_color = np.asarray([1, 1, 1])
        opt.point_size = 2.0

        view_ctrl = vis.get_view_control()
        if view_ctrl is None:
            raise RuntimeError("Failed to get view control")

    except Exception as exc:
        print(f"  Open3D offscreen rendering failed: {exc}")
        print("  Falling back to matplotlib...")
        _save_as_images_matplotlib(
            objects,
            scene_graph_geometries,
            output_dir,
            num_views,
            image_width,
            image_height,
            original_mesh_path,
        )
        return

    all_points_list: list[np.ndarray] = []
    for pcd in pcds:
        all_points_list.extend(np.asarray(pcd.points))
    all_points = np.array(all_points_list)
    center = np.mean(all_points, axis=0)
    scene_size = np.linalg.norm(np.max(all_points, axis=0) - np.min(all_points, axis=0))

    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for i, angle in enumerate(angles):
        distance = scene_size * 1.5
        eye = center + np.array(
            [
                distance * np.cos(angle),
                distance * np.sin(angle),
                scene_size * 0.5,
            ]
        )

        view_ctrl.set_lookat(center)
        view_ctrl.set_front((center - eye) / np.linalg.norm(center - eye))
        view_ctrl.set_up([0, 0, 1])
        view_ctrl.set_zoom(0.7)

        vis.poll_events()
        vis.update_renderer()

        image_path = images_dir / f"scene_graph_view_{i:02d}.png"
        vis.capture_screen_image(str(image_path), do_render=True)
        print(f"  Saved view {i + 1}/{num_views}: {image_path}")

    vis.destroy_window()
    print(f"\nAll images saved to: {images_dir}")


# -- PLY output ------------------------------------------------------


def _save_as_ply(
    objects: MapObjectList,
    scene_graph_geometries: list,
    output_dir: Path,
) -> None:
    """Save the scene as PLY point-cloud and mesh files."""
    print("\nSaving PLY files...")

    pcds = objects.get_values("pcd")

    combined_pcd = o3d.geometry.PointCloud()
    for pcd in pcds:
        combined_pcd += pcd
    combined_pcd = combined_pcd.voxel_down_sample(0.02)

    ply_dir = output_dir / "ply"
    ply_dir.mkdir(parents=True, exist_ok=True)

    pcd_path = ply_dir / "scene_pointcloud.ply"
    o3d.io.write_point_cloud(str(pcd_path), combined_pcd)
    print(f"  Point cloud saved: {pcd_path}")

    combined_mesh = o3d.geometry.TriangleMesh()
    for geom in scene_graph_geometries:
        if isinstance(geom, o3d.geometry.TriangleMesh):
            combined_mesh += geom

    if len(combined_mesh.vertices) > 0:
        mesh_path = ply_dir / "scene_graph.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), combined_mesh)
        print(f"  Scene-graph mesh saved: {mesh_path}")

    print(f"\nPLY files saved to: {ply_dir}")
    print("  Tip: open with MeshLab, CloudCompare, or Blender")


# -- HTML output -----------------------------------------------------


def _save_as_html(
    objects: MapObjectList,
    obj_centers: list[np.ndarray],
    edges: list[dict],
    class_colors: dict,
    output_dir: Path,
) -> None:
    """Save an interactive 3D HTML visualization."""
    print("\nGenerating HTML visualization...")

    try:
        import plotly.graph_objects as go  # noqa: F401

        use_plotly = True
    except ImportError:
        print("  Warning: plotly not installed, using matplotlib")
        use_plotly = False

    html_dir = output_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    classes = objects.get_most_common_class()
    colors_rgb = [class_colors[str(c)] for c in classes]

    if use_plotly:
        _save_html_plotly(objects, obj_centers, edges, colors_rgb, html_dir)
    else:
        _save_html_matplotlib(objects, obj_centers, edges, colors_rgb, html_dir)

    print(f"\nHTML visualization saved to: {html_dir}")
    print("  Tip: open the HTML file in a browser for interaction")


def _save_html_plotly(
    objects: MapObjectList,
    obj_centers: list[np.ndarray],
    edges: list[dict],
    colors_rgb: list[tuple],
    html_dir: Path,
) -> None:
    """Render interactive HTML with plotly."""
    import plotly.graph_objects as go

    fig = go.Figure()

    for i, (obj, color, _center) in enumerate(zip(objects, colors_rgb, obj_centers)):
        points = np.asarray(obj["pcd"].points)
        if len(points) > 1000:
            indices = np.random.choice(len(points), 1000, replace=False)
            points = points[indices]

        obj_classes = np.asarray(obj["class_id"])
        values, counts = np.unique(obj_classes, return_counts=True)
        obj_class = values[np.argmax(counts)]

        r, g, b = (
            int(color[0] * 255),
            int(color[1] * 255),
            int(color[2] * 255),
        )
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker={
                    "size": 2,
                    "color": f"rgb({r},{g},{b})",
                    "opacity": 0.8,
                },
                name=f"Object {i} (class {obj_class})",
                hovertext=f"Object ID: {i}<br>Class: {obj_class}",
            )
        )

    for edge in edges:
        if edge["object_relation"] == "none of these":
            continue
        id1 = edge["object1"]["id"]
        id2 = edge["object2"]["id"]
        relation = edge["object_relation"]

        fig.add_trace(
            go.Scatter3d(
                x=[obj_centers[id1][0], obj_centers[id2][0]],
                y=[obj_centers[id1][1], obj_centers[id2][1]],
                z=[obj_centers[id1][2], obj_centers[id2][2]],
                mode="lines",
                line={"color": "red", "width": 5},
                name=f"Relation: {relation}",
                hovertext=f"{id1} -> {id2}: {relation}",
                showlegend=False,
            )
        )

    centers_array = np.array(obj_centers)
    fig.add_trace(
        go.Scatter3d(
            x=centers_array[:, 0],
            y=centers_array[:, 1],
            z=centers_array[:, 2],
            mode="markers",
            marker={
                "size": 10,
                "color": "yellow",
                "symbol": "diamond",
                "line": {"color": "black", "width": 2},
            },
            name="Object centers",
            hovertext=[f"Object {i}" for i in range(len(obj_centers))],
        )
    )

    fig.update_layout(
        title="Scene Graph 3D Visualization (interactive)",
        scene={
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
            "aspectmode": "data",
        },
        width=1200,
        height=800,
        showlegend=True,
    )

    html_path = html_dir / "scene_graph_interactive.html"
    fig.write_html(str(html_path))
    print(f"  Interactive HTML saved: {html_path}")


def _save_html_matplotlib(
    objects: MapObjectList,
    obj_centers: list[np.ndarray],
    edges: list[dict],
    colors_rgb: list[tuple],
    html_dir: Path,
) -> None:
    """Render a static 3D image with matplotlib (plotly fallback)."""
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")

    for obj, color in zip(objects, colors_rgb):
        points = np.asarray(obj["pcd"].points)
        if len(points) > 500:
            indices = np.random.choice(len(points), 500, replace=False)
            points = points[indices]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=[color],
            s=1,
            alpha=0.6,
        )

    for edge in edges:
        if edge["object_relation"] == "none of these":
            continue
        id1 = edge["object1"]["id"]
        id2 = edge["object2"]["id"]
        ax.plot(
            [obj_centers[id1][0], obj_centers[id2][0]],
            [obj_centers[id1][1], obj_centers[id2][1]],
            [obj_centers[id1][2], obj_centers[id2][2]],
            "r-",
            linewidth=2,
        )

    centers_array = np.array(obj_centers)
    ax.scatter(
        centers_array[:, 0],
        centers_array[:, 1],
        centers_array[:, 2],
        c="yellow",
        s=100,
        marker="D",
        edgecolors="black",
        linewidths=2,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Scene Graph 3D Visualization")

    static_path = html_dir / "scene_graph_static.png"
    plt.savefig(static_path, dpi=150, bbox_inches="tight")
    print(f"  Static image saved: {static_path}")
    plt.close()


# -- main entry ------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    """Run the offscreen visualization pipeline."""
    print("=" * 80)
    print("Offscreen Scene-Graph Visualization")
    print("=" * 80)

    result_path = Path(args.result_path)
    output_dir = Path(args.output_dir)

    if not result_path.exists():
        print(f"Error: scene-graph file not found: {result_path}")
        sys.exit(1)

    if args.edge_file and not Path(args.edge_file).exists():
        print(f"Error: edge file not found: {args.edge_file}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    print(f"\nLoading scene graph: {result_path}")
    objects, bg_objects, class_colors = _load_result(result_path)
    print(f"  Loaded {len(objects)} objects")

    print("\nDownsampling point clouds...")
    for i in range(len(objects)):
        objects[i]["pcd"] = objects[i]["pcd"].voxel_down_sample(0.05)
    print("  Done")

    scene_graph_geometries: list = []
    obj_centers: list[np.ndarray] = []
    edges: list[dict] = []

    if args.edge_file:
        edge_path = Path(args.edge_file)
        print(f"\nLoading object relations: {edge_path}")
        with edge_path.open() as f:
            edges = json.load(f)
        print(f"  Loaded {len(edges)} relations")

        scene_graph_geometries, obj_centers = _create_scene_graph_geometries(
            objects, edges, class_colors
        )
        print("  Created scene-graph geometries")

    if args.output_format in ("images", "all"):
        _save_as_images(
            objects,
            bg_objects,
            scene_graph_geometries,
            output_dir,
            args.image_width,
            args.image_height,
            args.num_views,
            original_mesh_path=args.original_mesh,
        )

    if args.output_format in ("ply", "all"):
        _save_as_ply(objects, scene_graph_geometries, output_dir)

    if args.output_format in ("html", "all") and args.edge_file:
        _save_as_html(objects, obj_centers, edges, class_colors, output_dir)

    print("\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)
    print(f"\nAll output files saved in: {output_dir}")

    summary_path = output_dir / "summary.txt"
    with summary_path.open("w") as f:
        f.write("Scene-Graph Visualization Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write("Input files:\n")
        f.write(f"  - Scene graph: {args.result_path}\n")
        f.write(f"  - Relations: {args.edge_file}\n\n")
        f.write("Statistics:\n")
        f.write(f"  - Object count: {len(objects)}\n")
        f.write(f"  - Relation count: {len(edges)}\n\n")
        f.write(f"Output format: {args.output_format}\n")
        f.write(f"Output directory: {output_dir}\n")

    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    parser = _get_parser()
    main(parser.parse_args())

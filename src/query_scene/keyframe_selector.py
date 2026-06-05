"""
Keyframe Selector: Query-driven keyframe selection for VLM grounding.

This module implements the core functionality of selecting relevant keyframes
(RGB images) given a natural language query about a 3D scene.

Key Features:
1. Scene-aware query parsing: LLM maps query terms to scene object labels
2. CLIP-based semantic retrieval: Handles synonyms and semantic similarity
3. Visibility-based view selection: Finds views that best observe target objects
4. Joint coverage optimization: Selects views covering both target and anchor objects
5. Nested spatial query support: Complex queries like "pillow on sofa nearest door"
6. Multi-hypothesis fallback: DIRECT → PROXY → CONTEXT for robust grounding

Usage:
    selector = KeyframeSelector.from_scene_path("/path/to/scene", llm_model="gemini-2.5-pro")
    result = selector.select_keyframes_v2("the pillow on the sofa nearest the door", k=3)

    # result.keyframe_indices: [42, 67, 89]
    # result.keyframe_paths: [Path("frame000042.jpg"), ...]
    # result.target_objects: [ObjectNode(...), ...]
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import threading
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

# CLIP dependencies are intentionally lazy-loaded. Importing torch/open_clip at
# module load time costs hundreds of MB per benchmark worker even when the
# string/category path succeeds and CLIP fallback is never used.
open_clip = None
torch = None
HAS_CLIP = (
    importlib.util.find_spec("open_clip") is not None
    and importlib.util.find_spec("torch") is not None
)

# Import nested query modules
from .core import (
    DIRECTIONAL_RELATIONS,
    ExecutionPolicy,
    GroundingQuery,
    HypothesisKind,
    HypothesisOutputV1,
    ParseMode,
    QueryHypothesis,
    QueryNode,
    ReferenceFrame,
    SelectConstraint,
    SpatialConstraint,
)
from .lightweight_conceptgraph import load_conceptgraph_objects
from .parsing import QueryParser
from .query_executor import ExecutionMode, ExecutionResult, QueryExecutor
from .retrieval.spatial_checker import SpatialRelationChecker


@dataclass
class SceneObject:
    """Scene object representation with all attributes from pkl.gz file.

    Attributes are based on the output of:
    - 1b_extract_2d_segmentation_detect.sh (2D segmentation)
    - 2b_build_3d_object_map_detect.sh (3D object map)
    """

    # Core identification
    obj_id: int
    category: str  # Primary category (most common from class_name list)
    object_tag: str = ""  # Display tag (same as category if not specified)
    centroid: np.ndarray | None = None  # 3D position [x, y, z], computed from pcd_np

    # 3D point cloud data
    pcd_np: np.ndarray | None = None  # Point cloud (N, 3)
    pcd_color_np: np.ndarray | None = None  # Point colors (N, 3)
    bbox_np: np.ndarray | None = None  # 3D bounding box corners (8, 3)

    # CLIP features
    clip_ft: np.ndarray | None = None  # CLIP visual feature (1024,)
    text_ft: np.ndarray | None = None  # CLIP text feature (1024,)

    # Detection data (per-frame lists)
    image_idx: list[int] = field(default_factory=list)  # Frame indices where detected
    mask_idx: list[int] = field(default_factory=list)  # Mask index per frame
    class_name: list[str] = field(default_factory=list)  # Class name per detection
    class_id: list[int] = field(default_factory=list)  # Class ID per detection
    conf: list[float] = field(default_factory=list)  # Detection confidence
    xyxy: list[Any] = field(default_factory=list)  # 2D bounding boxes
    n_points: list[int] = field(default_factory=list)  # Points per detection
    pixel_area: list[int] = field(default_factory=list)  # Pixel area per detection

    # Metadata
    num_detections: int = 0  # Total detection count
    inst_color: np.ndarray | None = None  # Instance color (3,)
    is_background: bool = False

    # Optional rich information (from affordance extraction)
    summary: str = ""  # Description
    affordance_category: str = ""  # e.g., "lighting", "seating"
    co_objects: list[str] = field(default_factory=list)  # Related objects
    affordances: dict[str, Any] = field(default_factory=dict)  # Full affordance payload

    # Alias for backward compatibility
    @property
    def clip_feature(self) -> np.ndarray | None:
        return self.clip_ft

    @property
    def point_cloud(self) -> np.ndarray | None:
        return self.pcd_np

    @classmethod
    def from_dict(cls, obj_id: int, data: dict[str, Any]) -> SceneObject:
        """Create SceneObject from raw pkl.gz dict data."""
        from collections import Counter

        # Extract category (most common class_name)
        class_names = data.get("class_name", [])
        if class_names:
            valid_names = [
                n
                for n in class_names
                if n and str(n).lower() not in ("item", "none", "")
            ]
            if valid_names:
                category = Counter(valid_names).most_common(1)[0][0]
            else:
                category = class_names[0] if class_names else f"object_{obj_id}"
        else:
            category = f"object_{obj_id}"

        # Get point cloud and compute centroid
        pcd_np = data.get("pcd_np")
        centroid = None
        if pcd_np is not None and len(pcd_np) > 0:
            pcd_np = np.asarray(pcd_np)
            centroid = pcd_np.mean(axis=0)

        return cls(
            obj_id=obj_id,
            category=category,
            object_tag=category,
            centroid=centroid,
            # 3D data
            pcd_np=pcd_np,
            pcd_color_np=(
                np.asarray(data["pcd_color_np"])
                if data.get("pcd_color_np") is not None
                else None
            ),
            bbox_np=(
                np.asarray(data["bbox_np"]) if data.get("bbox_np") is not None else None
            ),
            # Features
            clip_ft=(
                np.asarray(data["clip_ft"]) if data.get("clip_ft") is not None else None
            ),
            text_ft=(
                np.asarray(data["text_ft"]) if data.get("text_ft") is not None else None
            ),
            # Detection data
            image_idx=list(data.get("image_idx", [])),
            mask_idx=list(data.get("mask_idx", [])),
            class_name=list(data.get("class_name", [])),
            class_id=list(data.get("class_id", [])),
            conf=list(data.get("conf", [])),
            xyxy=list(data.get("xyxy", [])),
            n_points=list(data.get("n_points", [])),
            pixel_area=list(data.get("pixel_area", [])),
            # Metadata
            num_detections=data.get("num_detections", 0),
            inst_color=(
                np.asarray(data["inst_color"])
                if data.get("inst_color") is not None
                else None
            ),
            is_background=bool(data.get("is_background", False)),
        )


@dataclass
class KeyframeResult:
    """Result of keyframe selection."""

    query: str
    target_term: str
    anchor_term: str | None

    # Selected keyframes
    keyframe_indices: list[int]  # Frame indices (view_id)
    keyframe_paths: list[Path]  # Paths to RGB images

    # Matched objects
    target_objects: list[SceneObject]
    anchor_objects: list[SceneObject]

    # Scores and metadata
    selection_scores: dict[int, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"Query: {self.query}",
            f"Target: '{self.target_term}' -> {len(self.target_objects)} objects",
        ]
        if self.anchor_term:
            lines.append(
                f"Anchor: '{self.anchor_term}' -> {len(self.anchor_objects)} objects"
            )
        lines.append(
            f"Selected {len(self.keyframe_indices)} keyframes: {self.keyframe_indices}"
        )
        return "\n".join(lines)


class KeyframeSelector:
    """Query-driven keyframe selector for VLM grounding.

    This class encapsulates the complete pipeline from query to keyframe selection:
    1. Load scene data (objects, poses, images)
    2. Build object-view visibility index
    3. Parse query with scene context
    4. Select optimal keyframes
    """

    def __init__(
        self,
        scene_path: Path,
        pcd_file: Path | None = None,
        affordance_file: Path | None = None,
        stride: int = 5,
        llm_model: str = None,
        use_pool: bool | None = None,
        prefer_lightweight_pcd: bool = True,
        ensure_lightweight_pcd: bool = False,
    ):
        """Initialize keyframe selector.

        Args:
            scene_path: Root path of the scene
            pcd_file: Path to .pkl.gz file with 3D objects
            affordance_file: Path to object_affordances.json (optional)
            stride: Frame stride used during mapping
            llm_model: LLM model name (e.g., "gemini-2.5-pro")
            use_pool: Whether to use Gemini pool. ``None`` auto-enables the
                pool for ``gemini-2.5-pro``.
        """
        self.scene_path = Path(scene_path)
        self.stride = stride
        self.llm_model = llm_model
        self.prefer_lightweight_pcd = prefer_lightweight_pcd
        self.ensure_lightweight_pcd = ensure_lightweight_pcd
        self.use_pool = (
            llm_model is not None and llm_model.strip().lower() == "gemini-2.5-pro"
            if use_pool is None
            else use_pool
        )

        # Data containers
        self.objects: list[SceneObject] = []
        self.object_features: np.ndarray | None = None  # (N, D)
        self.camera_poses: list[np.ndarray] = []  # List of 4x4 matrices
        self.image_paths: list[Path] = []
        self.depth_paths: list[Path] = []
        self.intrinsics: np.ndarray = np.array(
            [
                [600.0, 0, 599.5],
                [0, 600.0, 339.5],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Category index for query parsing
        self.scene_categories: list[str] = []

        # Bidirectional visibility index
        # object_to_views: obj_id -> [(view_id, score), ...] sorted by score desc
        self.object_to_views: dict[int, list[tuple[int, float]]] = {}
        # view_to_objects: view_id -> [(obj_id, score), ...] sorted by score desc
        self.view_to_objects: dict[int, list[tuple[int, float]]] = {}
        self.pose_velocities = np.array([], dtype=np.float64)
        self.pose_turn_rates = np.array([], dtype=np.float64)
        self.dwell_score = np.array([], dtype=np.float64)
        self.turn_score = np.array([], dtype=np.float64)
        self._K: np.ndarray | None = None
        self._img_wh: tuple[int, int] | None = None
        self._depth_cache: dict[int, np.ndarray] = {}
        self.pose_aware_enabled = False

        # CLIP model (lazy loaded)
        self._clip_model = None
        self._clip_tokenizer = None
        self._torch = None
        self._clip_model_lock = threading.Lock()

        # Query parsing components (lazy loaded)
        self._query_parser: QueryParser | None = None
        self._query_executor: QueryExecutor | None = None
        self._relation_checker: SpatialRelationChecker | None = None

        # Load data
        self._load_scene(pcd_file, affordance_file)
        self._load_or_build_visibility_index()

    @classmethod
    def from_scene_path(
        cls,
        scene_path: str,
        stride: int = 5,
        **kwargs,
    ) -> KeyframeSelector:
        """Create selector from scene path, auto-detecting files.

        Args:
            scene_path: Path to scene directory
            stride: Frame stride
            **kwargs: Additional arguments passed to __init__

        Returns:
            Initialized KeyframeSelector
        """
        scene_path = Path(scene_path)

        # Auto-detect PCD file
        pcd_dir = scene_path / "pcd_saves"
        pcd_files = list(pcd_dir.glob("*ram*_post.pkl.gz"))
        if not pcd_files:
            pcd_files = list(pcd_dir.glob("*_post.pkl.gz"))
        if not pcd_files:
            pcd_files = list(pcd_dir.glob("*.pkl.gz"))

        pcd_file = pcd_files[0] if pcd_files else None
        logger.info(f"Auto-detected PCD file: {pcd_file}")

        # Auto-detect affordance file
        affordance_file = scene_path / "sg_cache_detect" / "object_affordances.json"
        if not affordance_file.exists():
            affordance_file = scene_path / "sg_cache" / "object_affordances.json"
        if not affordance_file.exists():
            affordance_file = None
            logger.warning("No affordance file found, using basic object info")
        else:
            logger.info(f"Using affordance file: {affordance_file}")

        return cls(scene_path, pcd_file, affordance_file, stride, **kwargs)

    def _load_scene(self, pcd_file: Path | None, affordance_file: Path | None) -> None:
        """Load scene data from files."""
        logger.info(f"Loading scene from: {self.scene_path}")

        # Load 3D objects from PCD file
        if pcd_file and pcd_file.exists():
            self._load_objects_from_pcd(pcd_file)

        # Enrich with affordance data if available
        if affordance_file and affordance_file.exists():
            self._load_affordances(affordance_file)

        # Enrich with LLM-generated metadata (enriched_objects.json)
        enrichment_file = self.scene_path / "enriched_objects.json"
        if not enrichment_file.exists():
            raise FileNotFoundError(
                f"Enrichment file not found: {enrichment_file}. "
                f"Run `python -m src.scripts.enrich_objects` first."
            )
        self._load_enrichment(enrichment_file)

        # Load camera poses
        self._load_camera_poses()
        if len(self.camera_poses) >= 2:
            self._compute_trajectory_stats()
        else:
            num_poses = len(self.camera_poses)
            self.pose_velocities = np.zeros(num_poses, dtype=np.float32)
            self.pose_turn_rates = np.zeros(num_poses, dtype=np.float32)
            self.dwell_score = np.zeros(num_poses, dtype=np.float32)
            self.turn_score = np.zeros(num_poses, dtype=np.float32)
            logger.warning(
                "[KeyframeSelector] degenerate trajectory (len={} poses); "
                "pose-aware features disabled for this scene",
                num_poses,
            )

        # Set image paths
        self._set_image_paths()

        # Build category index (multi-label: includes minority detection classes)
        self.scene_categories = self._build_multilabel_categories()

        logger.success(
            f"Loaded {len(self.objects)} objects, {len(self.camera_poses)} poses"
        )
        logger.info(f"Scene categories: {self.scene_categories[:20]}...")

    def _load_objects_from_pcd(self, pcd_file: Path) -> None:
        """Load objects from ConceptGraphs PCD file."""
        raw_objects = load_conceptgraph_objects(
            pcd_file,
            prefer_lightweight=self.prefer_lightweight_pcd,
            ensure_lightweight=self.ensure_lightweight_pcd,
        )
        features = []

        for i, obj in enumerate(raw_objects):
            # Get centroid from the lightweight cache when possible. The raw
            # ConceptGraph pkl also contains per-detection masks that can expand
            # to many GB during pickle.load, so high-concurrency prep should use
            # the stripped sidecar cache and avoid requiring pcd_np here.
            pcd_np = obj.get("pcd_np")
            bbox_np_raw = obj.get("bbox_np")
            centroid_raw = obj.get("centroid")
            if pcd_np is not None and len(pcd_np) > 0:
                pcd_np = np.asarray(pcd_np, dtype=np.float32)
                centroid = pcd_np.mean(axis=0)
            elif centroid_raw is not None:
                centroid_arr = np.asarray(centroid_raw, dtype=np.float32).reshape(-1)
                if centroid_arr.size < 3:
                    continue
                centroid = centroid_arr[:3]
                pcd_np = None
            elif bbox_np_raw is not None:
                bbox_for_centroid = np.asarray(bbox_np_raw, dtype=np.float32)
                if bbox_for_centroid.size == 0 or bbox_for_centroid.shape[-1] < 3:
                    continue
                centroid = bbox_for_centroid.reshape(-1, bbox_for_centroid.shape[-1])[
                    :, :3
                ].mean(axis=0)
                pcd_np = None
            else:
                continue

            # Get CLIP feature
            clip_ft = obj.get("clip_ft")
            if clip_ft is not None:
                if hasattr(clip_ft, "cpu"):
                    clip_ft = clip_ft.cpu().numpy()
                clip_ft = np.asarray(clip_ft, dtype=np.float32).flatten()
                features.append(clip_ft)
            else:
                features.append(None)

            # Get category from class_name list
            class_names = obj.get("class_name", [])
            if class_names:
                valid = [
                    n for n in class_names if n and n.lower() not in ["item", "object"]
                ]
                category = (
                    Counter(valid).most_common(1)[0][0] if valid else class_names[0]
                )
            else:
                category = f"object_{i}"

            # Get detection data for visibility scoring
            image_idx = obj.get("image_idx", [])
            xyxy = obj.get("xyxy", [])

            # Keep bbox for downstream VG proposal/candidate formatting.
            bbox_np_arr = (
                np.asarray(bbox_np_raw, dtype=np.float32)
                if bbox_np_raw is not None
                else None
            )

            scene_obj = SceneObject(
                obj_id=len(self.objects),
                category=category,
                centroid=centroid,
                pcd_np=pcd_np,
                bbox_np=bbox_np_arr,
                clip_ft=clip_ft,
                image_idx=image_idx,
                xyxy=xyxy,
                class_name=class_names,
            )
            self.objects.append(scene_obj)

        # Build CLIP feature matrix aligned with self.objects indices.
        # Missing features keep zero vectors to preserve index consistency.
        valid_features = [f for f in features if f is not None]
        if valid_features:
            feat_dim = int(valid_features[0].shape[0])
            aligned = np.zeros((len(features), feat_dim), dtype=np.float32)
            for idx, feat in enumerate(features):
                if feat is None:
                    continue
                if feat.shape[0] != feat_dim:
                    logger.warning(
                        f"Object {idx} clip_ft dimension mismatch: "
                        f"{feat.shape[0]} vs expected {feat_dim}, skipping"
                    )
                    continue
                aligned[idx] = feat

            # L2 normalize non-zero rows only
            norms = np.linalg.norm(aligned, axis=1, keepdims=True)
            non_zero = norms.squeeze(-1) > 0
            aligned[non_zero] = aligned[non_zero] / (norms[non_zero] + 1e-8)
            self.object_features = aligned

    # Labels that are verbs, adjectives, abstract terms, or body parts —
    # not useful as object categories for scene graph retrieval.
    _NOISE_LABELS: frozenset[str] = frozenset(
        {
            "sit",
            "lay",
            "hang",
            "open",
            "fill",
            "push",
            "lead to",
            "take",
            "walk",
            "attach",
            "hide",
            "slide",
            "curl",
            "sleep",
            "stand",
            "make",
            "wrap",
            "sew",
            "lead",
            "roll",
            "draw",
            "writing",
            "mark",
            "couple",
            "selfie",
            "peak",
            "shine",
            "comfort",
            "mess",
            "stuff",
            "dormitory",
            "camouflage",
            "flat",
            "twin",
            "back",
            "half",
            "man",
            "woman",
            "person",
            "head",
            "nose",
            "hand",
            "foot",
            "face",
            "item",
            "object",
        }
    )

    def _build_multilabel_categories(self) -> list[str]:
        """Build scene categories including minority detection labels.

        Unlike the previous approach that only kept the majority-vote primary
        category, this method includes ALL class names detected >= 2 times
        for each 3D object. This prevents information loss where e.g. an
        object with 6 "sink" and 4 "lamp" detections would previously drop
        "lamp" entirely.
        """
        all_cats: set[str] = set()
        for obj in self.objects:
            # Always include the primary category
            primary = obj.object_tag if obj.object_tag else obj.category
            if primary and primary.lower() not in self._NOISE_LABELS:
                all_cats.add(primary)

            # Include minority detection labels with count >= 2
            if obj.class_name:
                counts = Counter(obj.class_name)
                for cls, cnt in counts.items():
                    if cnt >= 2 and cls and cls.lower() not in self._NOISE_LABELS:
                        all_cats.add(cls)

        return sorted(all_cats)

    def _load_affordances(self, affordance_file: Path) -> None:
        """Load and merge affordance data."""
        with open(affordance_file) as f:
            affordances = json.load(f)

        # Create mapping by ID
        aff_by_id = {a["id"]: a for a in affordances}

        for obj in self.objects:
            if obj.obj_id in aff_by_id:
                aff = aff_by_id[obj.obj_id]
                obj.object_tag = aff.get("object_tag", obj.object_tag)
                if obj.object_tag:
                    obj.category = obj.object_tag
                obj.summary = aff.get("summary", obj.summary)
                obj.affordance_category = aff.get("category", obj.affordance_category)

                affs = aff.get("affordances", {})
                if isinstance(affs, dict):
                    obj.affordances = affs
                    obj.co_objects = affs.get("co_objects", obj.co_objects)

    def _load_enrichment(self, enrichment_file: Path) -> None:
        """Load LLM-generated object enrichment from enriched_objects.json."""
        with open(enrichment_file) as f:
            data = json.load(f)

        enrich_by_id = {
            int(entry["obj_id"]): entry
            for entry in data.get("objects", [])
            if entry.get("status") == "success"
        }

        enriched_count = 0
        for obj in self.objects:
            entry = enrich_by_id.get(obj.obj_id)
            if entry is None:
                continue
            enrichment = entry.get("enrichment", {})
            obj.category = enrichment.get("category", obj.category)
            obj.object_tag = obj.category
            obj.summary = enrichment.get("description", obj.summary)
            obj.affordance_category = enrichment.get(
                "usability", obj.affordance_category
            )
            obj.co_objects = enrichment.get("nearby_objects", obj.co_objects)
            obj.affordances = enrichment
            enriched_count += 1

        logger.info(
            f"Loaded enrichment for {enriched_count}/{len(self.objects)} objects "
            f"from {enrichment_file.name}"
        )

    # Search order for camera trajectory files. The first existing path is used.
    # Order is pack-authoritative -> raw -> conceptgraph for historical layouts.
    # Each entry is relative to ``self.scene_path``. We also probe one level
    # up (the scene root) because in NR3D layout ``scene_path`` is the
    # ``.../scene0629_00/conceptgraph`` subdir and the packs / raw / traj.txt
    # actually live as siblings of conceptgraph, i.e. one level up.
    _TRAJ_SEARCH_RELATIVE: tuple[str, ...] = (
        "traj.txt",
        "../raw/traj.txt",
        "../conceptgraph/traj.txt",
        "../traj.txt",
    )

    def _candidate_pack_roots(self) -> list[Path]:
        """Return scene roots that may contain ``pack_*`` subdirs.

        OpenEQA layout: scene_path = .../scene0011_00, packs live under
        scene_path. NR3D layout: scene_path = .../scene0629_00/conceptgraph,
        packs live under scene_path.parent (i.e. .../scene0629_00).
        """
        roots: list[Path] = []
        if self.scene_path.exists():
            roots.append(self.scene_path)
            parent = self.scene_path.parent
            if parent.exists() and parent != self.scene_path:
                roots.append(parent)
        return roots

    def _describe_trajectory_search_paths(self) -> list[str]:
        """Listing used by error messages and tests."""
        paths = [
            str((self.scene_path / rel).resolve())
            for rel in self._TRAJ_SEARCH_RELATIVE
        ]
        for root in self._candidate_pack_roots():
            paths.append(str(root / "pack_*/camera_trajectory.json"))
        return paths

    def _find_camera_trajectory_file(self) -> Path | None:
        """Probe known trajectory locations and return the first existing path.

        Order:
        1. pack/camera_trajectory.json under either ``scene_path`` (OpenEQA)
           or ``scene_path.parent`` (NR3D, where ``scene_path`` is the
           ``conceptgraph`` subdir).
        2. ``traj.txt`` under ``scene_path``, ``../raw/``, ``../conceptgraph/``,
           or ``../`` (scene root).

        Existence is necessary but not sufficient - the caller still has to
        verify the parsed payload contains usable 4x4 poses (see
        ``_load_camera_poses``), otherwise we keep probing.
        """
        for root in self._candidate_pack_roots():
            for pack_dir in sorted(root.glob("pack_*")):
                candidate = pack_dir / "camera_trajectory.json"
                if candidate.is_file():
                    return candidate

        for rel in self._TRAJ_SEARCH_RELATIVE:
            candidate = (self.scene_path / rel).resolve()
            if candidate.exists():
                return candidate

        return None

    def _enumerate_camera_trajectory_files(self) -> list[Path]:
        """Return every existing trajectory candidate in search order.

        Unlike ``_find_camera_trajectory_file``, this returns the full list so
        ``_load_camera_poses`` can fall through to the next source when a
        preferred file parses to zero usable 4x4 poses. All paths are
        canonicalised via ``resolve()`` so the dedup set works correctly on
        macOS where ``/var`` is a symlink to ``/private/var`` and glob /
        Path.resolve disagree on the prefix.
        """
        found: list[Path] = []
        seen: set[Path] = set()

        def _add(candidate: Path) -> None:
            if not candidate.exists():
                return
            canonical = candidate.resolve()
            if canonical in seen:
                return
            seen.add(canonical)
            found.append(canonical)

        for root in self._candidate_pack_roots():
            for pack_dir in sorted(root.glob("pack_*")):
                _add(pack_dir / "camera_trajectory.json")

        for rel in self._TRAJ_SEARCH_RELATIVE:
            _add((self.scene_path / rel).resolve())

        return found

    def _load_camera_poses(self) -> None:
        """Load camera poses from the first source that yields usable 4x4 poses.

        Probes pack JSON then traj.txt variants (see
        ``_enumerate_camera_trajectory_files``). For each candidate we parse
        and require at least one full 4x4 pose; if the candidate yields zero
        (e.g. NR3D pack JSON which carries position-only entries) we move on
        to the next candidate and log the rejection - we DO NOT silently
        accept an empty result and suppress the next fallback.

        If no candidate yields any 4x4 poses we leave ``self.camera_poses``
        empty. The strict-no-fallback raise for viewer-frame requests lives
        in ``_guard_viewpoint_traj_available``; this loader stays
        best-effort so callers that don't need viewer-frame geometry are
        unaffected.
        """
        self.camera_trajectory_path = None
        candidates = self._enumerate_camera_trajectory_files()
        if not candidates:
            logger.warning(
                "[KeyframeSelector] No camera trajectory found in any of: "
                f"{self._describe_trajectory_search_paths()}"
            )
            return

        for traj_path in candidates:
            try:
                if traj_path.suffix.lower() == ".json":
                    all_poses = self._read_camera_trajectory_json(traj_path)
                else:
                    all_poses = self._read_camera_trajectory_traj_txt(traj_path)
            except Exception as exc:
                logger.error(
                    f"[KeyframeSelector] Failed to parse trajectory file "
                    f"{traj_path}: {exc}; trying next source."
                )
                continue

            if not all_poses:
                logger.warning(
                    f"[KeyframeSelector] {traj_path} yielded zero usable 4x4 "
                    "poses (likely a position-only payload); trying next source."
                )
                continue

            self.camera_trajectory_path = traj_path
            self.camera_poses = [
                all_poses[i]
                for i in range(0, len(all_poses), self.stride)
                if i < len(all_poses)
            ]
            logger.info(
                f"[KeyframeSelector] Loaded {len(self.camera_poses)} camera "
                f"poses from {traj_path}"
            )
            return

        logger.warning(
            "[KeyframeSelector] None of the camera trajectory candidates "
            f"yielded usable 4x4 poses. Tried: {[str(p) for p in candidates]}"
        )

    @staticmethod
    def _read_camera_trajectory_traj_txt(traj_path: Path) -> list[np.ndarray]:
        """Parse traj.txt: either 16-numbers-per-line or 4-lines-per-matrix."""
        with open(traj_path) as f:
            lines = f.readlines()
        first_line_nums = len(lines[0].split()) if lines else 0

        all_poses: list[np.ndarray] = []
        if first_line_nums == 16:
            for line in lines:
                nums = [float(x) for x in line.split()]
                if len(nums) == 16:
                    all_poses.append(np.array(nums).reshape(4, 4))
        else:
            for i in range(0, len(lines), 4):
                if i + 4 <= len(lines):
                    try:
                        pose = np.array(
                            [
                                [float(x) for x in lines[i].split()],
                                [float(x) for x in lines[i + 1].split()],
                                [float(x) for x in lines[i + 2].split()],
                                [float(x) for x in lines[i + 3].split()],
                            ]
                        )
                        all_poses.append(pose)
                    except (ValueError, IndexError) as exc:
                        logger.warning(
                            f"[KeyframeSelector] Malformed pose at line {i} "
                            f"in {traj_path}: {exc}"
                        )
                        continue
        return all_poses

    @staticmethod
    def _read_camera_trajectory_json(traj_path: Path) -> list[np.ndarray]:
        """Parse pack-style camera_trajectory.json into a list of 4x4 poses.

        Supported payloads:
        - Top-level list of 4x4 matrices (16 numbers or 4x4 nested list).
        - Top-level dict with ``poses`` / ``camera_poses`` / ``frames`` whose
          value is a list of matrices or a list of dicts with ``world_T_cam``
          / ``pose`` / ``matrix`` / ``world_to_cam`` fields.

        Explicitly rejected payloads (returns empty list, logs at WARNING):
        - Position-only dict of integer-keyed XYZ entries, e.g.
          ``{"0": [x,y,z], "1": [x,y,z], ...}``. This is the NR3D pack
          format; it lacks orientation so it cannot be used for viewer-frame
          pose resolution. Returning [] lets ``_load_camera_poses`` fall
          through to the next source (typically raw/traj.txt with full 4x4).

        Other malformed entries are skipped individually with a debug log.
        """
        with open(traj_path) as f:
            payload = json.load(f)

        entries: list = []
        if isinstance(payload, list):
            entries = payload
        elif isinstance(payload, dict):
            named = (
                payload.get("poses")
                or payload.get("camera_poses")
                or payload.get("frames")
            )
            if named is not None:
                entries = named
            elif KeyframeSelector._looks_like_position_only_dict(payload):
                logger.warning(
                    f"[KeyframeSelector] {traj_path} is a position-only dict "
                    "(integer-keyed XYZ entries). Viewer-frame geometry needs "
                    "full 4x4 poses; rejecting this source so the loader falls "
                    "through to traj.txt."
                )
                return []
            else:
                raise ValueError(
                    f"camera_trajectory.json at {traj_path} has dict payload "
                    "but no recognized poses/camera_poses/frames key and is not "
                    "a position-only dict; refusing to parse silently."
                )
        else:
            raise ValueError(
                f"camera_trajectory.json at {traj_path} has unexpected "
                f"top-level type {type(payload).__name__}"
            )

        all_poses: list[np.ndarray] = []
        for entry in entries:
            mat = entry
            if isinstance(entry, dict):
                mat = (
                    entry.get("world_T_cam")
                    or entry.get("pose")
                    or entry.get("matrix")
                    or entry.get("world_to_cam")
                )
            if mat is None:
                continue
            try:
                arr = np.asarray(mat, dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if arr.shape == (16,):
                arr = arr.reshape(4, 4)
            if arr.shape == (4, 4):
                all_poses.append(arr)
        return all_poses

    @staticmethod
    def _looks_like_position_only_dict(payload: dict) -> bool:
        """True iff payload is a dict whose values are all 3-element XYZ lists.

        Catches the NR3D pack format ``{"0": [x,y,z], "1": [x,y,z], ...}``.
        We do not try to upgrade it to a 4x4 because there is no orientation
        anywhere in the file; ``_load_camera_poses`` will fall through to
        a traj.txt source that does carry orientation.
        """
        if not payload:
            return False
        for k, v in payload.items():
            if not isinstance(k, str) or not k.isdigit():
                return False
            if not isinstance(v, list) or len(v) != 3:
                return False
            for x in v:
                if not isinstance(x, (int, float)):
                    return False
        return True

    def _compute_trajectory_stats(self) -> None:
        """Compute per-view translational and rotational trajectory saliency."""
        num_poses = len(self.camera_poses)
        if num_poses < 2:
            raise ValueError(f"Cannot compute trajectory stats with {num_poses} poses")

        translations = np.asarray(
            [pose[:3, 3] for pose in self.camera_poses],
            dtype=np.float64,
        )
        translation_steps = np.linalg.norm(np.diff(translations, axis=0), axis=1)
        velocities = np.empty(num_poses, dtype=np.float64)
        velocities[:-1] = translation_steps
        velocities[-1] = translation_steps[-1]

        forward_vectors = []
        for pose_index, pose in enumerate(self.camera_poses):
            try:
                forward_vectors.append(self._normalize_vector(-pose[:3, 2]))
            except ValueError:
                logger.warning(
                    "[KeyframeSelector] degenerate forward vector at pose {}; "
                    "treating as zero turn",
                    pose_index,
                )
                forward_vectors.append(np.array([0.0, 0.0, -1.0], dtype=np.float64))
        forward_vectors = np.asarray(forward_vectors, dtype=np.float64)
        cos_turn = np.sum(
            forward_vectors[:-1] * forward_vectors[1:],
            axis=1,
        )
        turn_steps = np.arccos(np.clip(cos_turn, -1.0, 1.0))
        turn_rates = np.empty(num_poses, dtype=np.float64)
        turn_rates[:-1] = turn_steps
        turn_rates[-1] = turn_steps[-1]

        smoothed_velocity = self._median_smooth_1d(velocities, radius=2)
        smoothed_turn = self._median_smooth_1d(turn_rates, radius=2)

        self.pose_velocities = velocities
        self.pose_turn_rates = turn_rates
        self.dwell_score = 1.0 / (1.0 + 5.0 * smoothed_velocity)
        turn_p95 = float(np.percentile(smoothed_turn, 95))
        if turn_p95 <= 1e-8:
            self.turn_score = np.zeros_like(smoothed_turn)
        else:
            self.turn_score = np.clip(smoothed_turn / turn_p95, 0.0, 1.0)

    @staticmethod
    def _median_smooth_1d(values: np.ndarray, radius: int = 2) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        smoothed = np.empty_like(values)
        for index in range(values.shape[0]):
            start = max(0, index - radius)
            end = min(values.shape[0], index + radius + 1)
            smoothed[index] = float(np.median(values[start:end]))
        return smoothed

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-8:
            raise ValueError("camera forward vector is degenerate")
        return np.asarray(vector, dtype=np.float64) / norm

    def _resolve_raw_dir(self) -> Path:
        """Return the raw scene directory with intrinsic and per-frame images."""
        overlay_candidate = self.scene_path / "raw"
        if overlay_candidate.exists():
            return overlay_candidate

        canonical = self.scene_path.parent / "raw"
        if canonical.exists():
            return canonical

        raise FileNotFoundError(
            f"raw dir not found at {overlay_candidate} or {canonical}"
        )

    def _get_intrinsic(self) -> tuple[np.ndarray, tuple[int, int]]:
        if self._K is None or self._img_wh is None:
            from .frustum import load_scene_intrinsic

            self._K, self._img_wh = load_scene_intrinsic(self._resolve_raw_dir())
        return self._K, self._img_wh

    def _load_depth_for_view(self, view_id: int) -> np.ndarray:
        """Load one raw depth map in meters and cache it in-memory."""
        if view_id in self._depth_cache:
            return self._depth_cache[view_id]
        if view_id < 0 or view_id >= len(self.camera_poses):
            raise ValueError(f"view_id {view_id} out of range")

        from PIL import Image

        frame_id = self.map_view_to_frame(view_id)
        depth_path = self._resolve_raw_dir() / f"{frame_id:06d}-depth.png"
        if not depth_path.exists():
            raise FileNotFoundError(depth_path)

        depth = np.asarray(Image.open(depth_path), dtype=np.float32) / 1000.0
        if depth.ndim != 2:
            raise ValueError(f"depth image must be HxW, got {depth.shape}")
        if not np.isfinite(depth).all():
            raise ValueError(f"depth image contains non-finite values: {depth_path}")

        self._depth_cache[view_id] = depth
        return depth

    def _set_image_paths(self) -> None:
        """Set paths to RGB and depth images.

        Checks multiple layouts:
        1. conceptgraph/results/frame*.jpg (OpenEQA/ConceptGraph standard)
        2. ../raw/*-rgb.jpg (EmbodiedScan prepared scenes)
        """
        results_dir = self.scene_path / "results"
        raw_dir = self.scene_path.parent / "raw"

        # Find all RGB images — try standard results/ first
        all_images = sorted(results_dir.glob("frame*.jpg"))
        if not all_images:
            all_images = sorted(results_dir.glob("*.jpg"))

        # Fallback: EmbodiedScan raw/ layout (*-rgb.jpg)
        if not all_images and raw_dir.is_dir():
            all_images = sorted(raw_dir.glob("*-rgb.jpg"))
            if not all_images:
                all_images = sorted(raw_dir.glob("*-rgb.png"))
            if all_images:
                logger.info(
                    "Using raw/ layout ({} images from {})",
                    len(all_images),
                    raw_dir,
                )

        # Apply stride
        self.image_paths = [
            all_images[i]
            for i in range(0, len(all_images), self.stride)
            if i < len(all_images)
        ]

        # Find depth images — try both layouts
        all_depths = sorted(results_dir.glob("depth*.png"))
        if not all_depths and raw_dir.is_dir():
            all_depths = sorted(raw_dir.glob("*-depth.png"))

        self.depth_paths = [
            all_depths[i]
            for i in range(0, len(all_depths), self.stride)
            if i < len(all_depths)
        ]

    def _load_or_build_visibility_index(self) -> None:
        """Load precomputed visibility index or build online.

        Prefers offline index from scene_path/indices/visibility_index.pkl.
        If the file exists but fails to load, raises instead of silently falling back.
        If the file does not exist, builds online with a warning.
        """
        index_path = self.scene_path / "indices" / "visibility_index.pkl"

        if index_path.exists():
            # File exists — must load successfully or raise
            self._load_visibility_index(index_path)
            return

        logger.warning(
            f"Visibility index not found at {index_path}. "
            "Building online (slower). Run 6b_build_visibility_index.sh for faster inference."
        )
        self._build_visibility_index_online()

    def _load_visibility_index(self, index_path: Path) -> None:
        """Load precomputed bidirectional visibility index from file."""
        logger.info(f"Loading visibility index from {index_path}")

        with open(index_path, "rb") as f:
            data = pickle.load(f)

        metadata = data.get("metadata", {})

        # Support both old format (visibility_index) and new format (object_to_views, view_to_objects)
        if "object_to_views" in data:
            # New bidirectional format
            raw_obj_to_views = data.get("object_to_views", {})
            raw_view_to_objs = data.get("view_to_objects", {})

            self.object_to_views = {int(k): v for k, v in raw_obj_to_views.items()}
            self.view_to_objects = {int(k): v for k, v in raw_view_to_objs.items()}
        else:
            # Old format - only object_to_views, build reverse index
            raw_index = data.get("visibility_index", {})
            self.object_to_views = {int(k): v for k, v in raw_index.items()}

            # Build reverse index
            self.view_to_objects = {}
            for obj_id, views in self.object_to_views.items():
                for view_id, score in views:
                    if view_id not in self.view_to_objects:
                        self.view_to_objects[view_id] = []
                    self.view_to_objects[view_id].append((obj_id, score))

            # Sort by score
            for view_id in self.view_to_objects:
                self.view_to_objects[view_id].sort(key=lambda x: x[1], reverse=True)

        # Check stride consistency
        saved_stride = metadata.get("stride", self.stride)
        if saved_stride != self.stride:
            logger.warning(
                f"Stride mismatch: saved={saved_stride}, current={self.stride}. "
                "View indices may be incorrect."
            )

        logger.success(
            f"Loaded bidirectional visibility index: "
            f"{len(self.object_to_views)} objects, {len(self.view_to_objects)} views"
        )

    def _build_visibility_index_online(self) -> None:
        """Build bidirectional visibility index online using detection ground truth."""
        logger.info("Building visibility index online (using detection data)...")

        img_width, img_height = 1200, 680
        img_area = img_width * img_height
        max_distance = 5.0

        self.view_to_objects = {}

        for obj in self.objects:
            if obj.image_idx:
                scores = self._compute_visibility_scores(
                    obj, img_width, img_height, img_area, max_distance
                )
            else:
                scores = self._compute_geometric_scores(obj, max_distance)

            scores.sort(key=lambda x: x[1], reverse=True)
            self.object_to_views[obj.obj_id] = scores

            for view_id, score in scores:
                if view_id not in self.view_to_objects:
                    self.view_to_objects[view_id] = []
                self.view_to_objects[view_id].append((obj.obj_id, score))

        for vid in self.view_to_objects:
            self.view_to_objects[vid].sort(key=lambda x: x[1], reverse=True)

        logger.success(
            f"Built index: {len(self.object_to_views)} objects, {len(self.view_to_objects)} views"
        )

    def _compute_visibility_scores(
        self, obj: SceneObject, img_w: int, img_h: int, img_area: int, max_dist: float
    ) -> list[tuple[int, float]]:
        """Compute visibility scores using detection data."""
        scores = []
        view_indices = {}
        for i, vid in enumerate(obj.image_idx):
            view_indices.setdefault(vid, []).append(i)

        for view_id, indices in view_indices.items():
            if view_id >= len(self.camera_poses):
                continue

            completeness = 0.0
            for idx in indices:
                if idx < len(obj.xyxy) and obj.xyxy[idx] is not None:
                    xyxy = obj.xyxy[idx]
                    if hasattr(xyxy, "__len__") and len(xyxy) == 4:
                        x1, y1, x2, y2 = xyxy
                        bbox_area = (x2 - x1) * (y2 - y1)
                        size_score = min(1.0, bbox_area / (img_area * 0.3))
                        is_clipped = (
                            x1 < 10 or y1 < 10 or x2 > img_w - 10 or y2 > img_h - 10
                        )
                        completeness = max(
                            completeness,
                            max(0, size_score - (0.3 if is_clipped else 0)),
                        )

            geo = 0.0
            if not np.allclose(obj.centroid, 0):
                pose = self.camera_poses[view_id]
                cam_pos = pose[:3, 3]
                dist = np.linalg.norm(obj.centroid - cam_pos)
                if dist <= max_dist:
                    dist_s = max(0, 1 - dist / max_dist)
                    view_dir = (obj.centroid - cam_pos) / (
                        np.linalg.norm(obj.centroid - cam_pos) + 1e-8
                    )
                    angle_s = max(0, np.dot(view_dir, -pose[:3, 2]))
                    geo = 0.6 * dist_s + 0.4 * angle_s

            quality = min(1.0, len(indices) / 3.0)
            scores.append((view_id, 0.5 * completeness + 0.3 * geo + 0.2 * quality))
        return scores

    def _compute_geometric_scores(
        self, obj: SceneObject, max_dist: float
    ) -> list[tuple[int, float]]:
        """Fallback: geometric-only scoring."""
        scores = []
        for view_id, pose in enumerate(self.camera_poses):
            cam_pos = pose[:3, 3]
            dist = np.linalg.norm(obj.centroid - cam_pos)
            if dist > max_dist:
                continue
            dist_s = max(0, 1 - dist / max_dist)
            view_dir = (obj.centroid - cam_pos) / (
                np.linalg.norm(obj.centroid - cam_pos) + 1e-8
            )
            angle_s = max(0, np.dot(view_dir, -pose[:3, 2]))
            combined = 0.6 * dist_s + 0.4 * angle_s
            if combined > 0.1:
                scores.append((view_id, combined))
        return scores

    def _compute_visibility_scores_from_detections(
        self,
        obj: SceneObject,
        img_width: int,
        img_height: int,
        img_area: int,
        max_distance: float,
    ) -> list[tuple[int, float]]:
        """Compute visibility scores using detection ground truth."""
        scores = []

        # Build view_id -> list of detection indices
        view_to_indices: dict[int, list[int]] = {}
        for i, vid in enumerate(obj.image_idx):
            if vid not in view_to_indices:
                view_to_indices[vid] = []
            view_to_indices[vid].append(i)

        for view_id, indices in view_to_indices.items():
            if view_id >= len(self.camera_poses):
                continue

            # 1. Completeness score (50% weight)
            best_completeness = 0.0
            for idx in indices:
                if idx < len(obj.xyxy) and obj.xyxy[idx] is not None:
                    xyxy = obj.xyxy[idx]
                    if hasattr(xyxy, "__len__") and len(xyxy) == 4:
                        x1, y1, x2, y2 = xyxy
                        bbox_w, bbox_h = x2 - x1, y2 - y1
                        bbox_area = bbox_w * bbox_h

                        # Size score
                        size_score = min(1.0, bbox_area / (img_area * 0.3))

                        # Clip penalty
                        margin = 10
                        is_clipped = (
                            x1 < margin
                            or y1 < margin
                            or x2 > img_width - margin
                            or y2 > img_height - margin
                        )
                        clip_penalty = 0.3 if is_clipped else 0.0

                        completeness = max(0, size_score - clip_penalty)
                        best_completeness = max(best_completeness, completeness)

            # 2. Geometric score (30% weight)
            geo_score = 0.0
            if not np.allclose(obj.centroid, 0):
                pose = self.camera_poses[view_id]
                cam_pos = pose[:3, 3]
                distance = np.linalg.norm(obj.centroid - cam_pos)
                if distance <= max_distance:
                    dist_score = max(0, 1 - distance / max_distance)
                    view_dir = obj.centroid - cam_pos
                    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)
                    cam_forward = -pose[:3, 2]
                    angle_score = max(0, np.dot(view_dir, cam_forward))
                    geo_score = 0.6 * dist_score + 0.4 * angle_score

            # 3. Detection quality (20% weight)
            quality_score = min(1.0, len(indices) / 3.0)

            # Combined score
            combined = 0.5 * best_completeness + 0.3 * geo_score + 0.2 * quality_score
            scores.append((view_id, float(combined)))

        return scores

    def _compute_visibility_scores_geometric(
        self, obj: SceneObject, max_distance: float
    ) -> list[tuple[int, float]]:
        """Fallback: compute visibility using only geometric scoring."""
        scores = []
        for view_id, pose in enumerate(self.camera_poses):
            cam_pos = pose[:3, 3]
            distance = np.linalg.norm(obj.centroid - cam_pos)
            if distance > max_distance:
                continue
            dist_score = max(0, 1 - distance / max_distance)
            view_dir = obj.centroid - cam_pos
            view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)
            cam_forward = -pose[:3, 2]
            angle_score = max(0, np.dot(view_dir, cam_forward))
            combined = 0.6 * dist_score + 0.4 * angle_score
            if combined > 0.1:
                scores.append((view_id, combined))
        return scores

    def _load_clip_model(self) -> None:
        """Load CLIP model for text encoding.

        Does nothing if open_clip is not installed (macOS dev without GPU).
        Raises on load failure when open_clip IS installed (broken env).
        """
        if self._clip_model is not None:
            return

        if not HAS_CLIP:
            # open_clip not installed — CLIP features unavailable (acceptable on macOS)
            return

        lock = getattr(self, "_clip_model_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._clip_model_lock = lock

        with lock:
            if self._clip_model is not None:
                return

            logger.info("Loading CLIP model...")
            global open_clip, torch
            if open_clip is None or torch is None:
                try:
                    import open_clip as open_clip_module
                    import torch as torch_module
                except ImportError as exc:
                    raise RuntimeError(
                        "open_clip and torch are required for CLIP fallback"
                    ) from exc
                open_clip = open_clip_module
                torch = torch_module

            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-H-14", "laion2b_s32b_b79k"
            )
            self._clip_model = model.eval()
            self._clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
            self._torch = torch

            if torch.cuda.is_available():
                self._clip_model = self._clip_model.cuda()

            logger.success("CLIP model loaded")

    def _encode_text(self, text: str) -> np.ndarray | None:
        """Encode text to CLIP feature.

        Returns None if CLIP is not available (open_clip not installed).
        Raises on encoding failure when CLIP IS loaded.
        """
        self._load_clip_model()

        if self._clip_model is None:
            return None

        tokens = self._clip_tokenizer([text])
        torch_module = self._torch or torch
        if torch_module is None:
            raise RuntimeError("CLIP model loaded without torch module")
        if torch_module.cuda.is_available():
            tokens = tokens.cuda()

        with torch_module.no_grad():
            feat = self._clip_model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.cpu().numpy().flatten()

    def find_objects(self, query_term: str, top_k: int = 10) -> list[SceneObject]:
        """Find objects matching a query term.

        Two-stage matching:
        1. Exact/fuzzy string match on object_tag or category
        2. CLIP semantic similarity fallback

        Args:
            query_term: Search term
            top_k: Maximum objects to return

        Returns:
            List of matching SceneObject
        """
        query_lower = query_term.lower()

        # Stage 1: String matching
        matches = []
        for obj in self.objects:
            tag = (obj.object_tag or obj.category).lower()
            # Exact substring match
            if query_lower in tag or tag in query_lower:
                matches.append((obj, 1.0))

        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            return [m[0] for m in matches[:top_k]]

        # Stage 2: CLIP semantic matching (skipped when open_clip not installed)
        if self.object_features is not None:
            query_feat = self._encode_text(query_term)
            if query_feat is None:
                logger.warning(
                    f"No objects found for '{query_term}' "
                    "(CLIP unavailable, string match exhausted)"
                )
                return []
            # Compute similarities
            similarities = self.object_features @ query_feat
            top_indices = np.argsort(-similarities)[:top_k]

            # Filter by minimum similarity
            min_sim = 0.2
            matches = [
                (self.objects[i], similarities[i])
                for i in top_indices
                if similarities[i] > min_sim
            ]

            if matches:
                logger.info(
                    f"CLIP matched '{query_term}' -> "
                    f"{[(m[0].object_tag or m[0].category, f'{m[1]:.2f}') for m in matches[:5]]}"
                )
                return [m[0] for m in matches]

        logger.warning(f"No objects found for '{query_term}'")
        return []

    def get_best_views_for_object(self, obj_id: int, top_k: int = 5) -> list[int]:
        """Get best view indices for a single object."""
        views = self.object_to_views.get(obj_id, [])
        return [v[0] for v in views[:top_k]]

    def get_joint_coverage_views(
        self,
        object_ids: list[int],
        max_views: int = 3,
        *,
        pose_aware: bool = False,
        frustum_overlap_threshold: float = 0.7,
        redundancy_penalty: float = 0.2,
        dwell_weight: float = 0.15,
        frustum_method: str = "l1",
    ) -> list[int]:
        """Greedy selection of views that maximize joint coverage of objects.

        Args:
            object_ids: Object IDs to cover
            max_views: Maximum views to select

        Returns:
            List of selected view indices
        """
        if not pose_aware:
            return self._get_joint_coverage_views_v14(object_ids, max_views=max_views)

        if not object_ids:
            return []

        self._validate_frustum_method(frustum_method)

        # Collect all candidate views
        candidate_views: set[int] = set()
        for obj_id in object_ids:
            for view_id, _ in self.object_to_views.get(obj_id, []):
                candidate_views.add(view_id)

        if not candidate_views:
            return []

        # Build view -> {obj_id: score} mapping
        view_scores: dict[int, dict[int, float]] = {}
        for obj_id in object_ids:
            for view_id, score in self.object_to_views.get(obj_id, []):
                if view_id not in view_scores:
                    view_scores[view_id] = {}
                view_scores[view_id][obj_id] = score

        # Greedy selection
        selected = []
        covered_quality = dict.fromkeys(object_ids, 0.0)

        for _ in range(max_views):
            best_view, best_gain = None, 0.0

            for view_id in candidate_views - set(selected):
                # Compute marginal gain
                base_gain = 0.0
                for obj_id in object_ids:
                    obj_score = view_scores.get(view_id, {}).get(obj_id, 0)
                    if obj_score > covered_quality[obj_id]:
                        base_gain += obj_score - covered_quality[obj_id]

                gain = self._apply_pose_aware_gain(
                    base_gain=base_gain,
                    view_id=view_id,
                    selected=selected,
                    frustum_overlap_threshold=frustum_overlap_threshold,
                    redundancy_penalty=redundancy_penalty,
                    dwell_weight=dwell_weight,
                    frustum_method=frustum_method,
                )

                if gain > best_gain:
                    best_gain, best_view = gain, view_id

            if best_view is None:
                break

            selected.append(best_view)

            # Update covered quality
            for obj_id in object_ids:
                obj_score = view_scores.get(best_view, {}).get(obj_id, 0)
                covered_quality[obj_id] = max(covered_quality[obj_id], obj_score)

        return selected

    def _get_joint_coverage_views_v14(
        self,
        object_ids: list[int],
        max_views: int = 3,
    ) -> list[int]:
        """Preserve the original visibility-only greedy selector exactly."""
        if not object_ids:
            return []

        candidate_views: set[int] = set()
        for obj_id in object_ids:
            for view_id, _ in self.object_to_views.get(obj_id, []):
                candidate_views.add(view_id)

        if not candidate_views:
            return []

        view_scores: dict[int, dict[int, float]] = {}
        for obj_id in object_ids:
            for view_id, score in self.object_to_views.get(obj_id, []):
                if view_id not in view_scores:
                    view_scores[view_id] = {}
                view_scores[view_id][obj_id] = score

        selected: list[int] = []
        covered_quality = dict.fromkeys(object_ids, 0.0)

        for _ in range(max_views):
            best_view, best_gain = None, 0.0

            for view_id in candidate_views - set(selected):
                gain = 0.0
                for obj_id in object_ids:
                    obj_score = view_scores.get(view_id, {}).get(obj_id, 0)
                    if obj_score > covered_quality[obj_id]:
                        gain += obj_score - covered_quality[obj_id]

                if gain > best_gain:
                    best_gain, best_view = gain, view_id

            if best_view is None:
                break

            selected.append(best_view)

            for obj_id in object_ids:
                obj_score = view_scores.get(best_view, {}).get(obj_id, 0)
                covered_quality[obj_id] = max(covered_quality[obj_id], obj_score)

        return selected

    @staticmethod
    def _validate_frustum_method(frustum_method: str) -> None:
        if frustum_method not in {"l1", "l2"}:
            raise ValueError(
                f"frustum_method must be 'l1' or 'l2', got {frustum_method!r}"
            )

    def _apply_pose_aware_gain(
        self,
        *,
        base_gain: float,
        view_id: int,
        selected: list[int],
        frustum_overlap_threshold: float,
        redundancy_penalty: float,
        dwell_weight: float,
        frustum_method: str,
    ) -> float:
        if base_gain <= 0.0:
            return 0.0

        gain = float(base_gain)
        if selected:
            overlaps = [
                self._compute_view_frustum_overlap(
                    anchor_view_id=view_id,
                    neighbor_view_id=selected_view,
                    frustum_method=frustum_method,
                )
                for selected_view in selected
            ]
            if any(overlap > frustum_overlap_threshold for overlap in overlaps):
                gain *= redundancy_penalty

        dwell = 0.0
        if 0 <= view_id < len(self.dwell_score):
            dwell = float(self.dwell_score[view_id])
        gain *= 1.0 + dwell_weight * dwell
        return gain

    def _compute_view_frustum_overlap(
        self,
        *,
        anchor_view_id: int,
        neighbor_view_id: int,
        frustum_method: str,
    ) -> float:
        self._validate_frustum_method(frustum_method)
        if anchor_view_id < 0 or anchor_view_id >= len(self.camera_poses):
            raise ValueError(f"anchor_view_id {anchor_view_id} out of range")
        if neighbor_view_id < 0 or neighbor_view_id >= len(self.camera_poses):
            raise ValueError(f"neighbor_view_id {neighbor_view_id} out of range")

        k, img_wh = self._get_intrinsic()
        pose_anchor = self.camera_poses[anchor_view_id]
        pose_neighbor = self.camera_poses[neighbor_view_id]

        if frustum_method == "l1":
            from .frustum import frustum_overlap_l1

            return frustum_overlap_l1(pose_anchor, pose_neighbor, k, img_wh)

        from .frustum import frustum_overlap_l2

        depth = self._load_depth_for_view(anchor_view_id)
        return frustum_overlap_l2(depth, pose_anchor, pose_neighbor, k, img_wh)

    def _pad_keyframes_to_minimum(
        self,
        selected: list[int],
        object_ids: list[int],
        min_count: int = 3,
        *,
        pose_aware: bool = False,
        frustum_overlap_threshold: float = 0.7,
        redundancy_penalty: float = 0.2,
        dwell_weight: float = 0.15,
        frustum_method: str = "l1",
    ) -> list[int]:
        """Pad keyframes using highest-visibility views if below min_count.

        Uses deterministic visibility-based selection from view_to_objects.
        """
        if not pose_aware:
            return self._pad_keyframes_to_minimum_v14(
                selected,
                object_ids,
                min_count=min_count,
            )

        if len(selected) >= min_count:
            return selected

        self._validate_frustum_method(frustum_method)
        selected = list(selected)

        # Collect candidate views from target objects
        candidate_views: set[int] = set()
        for obj_id in object_ids:
            for view_id, _ in self.object_to_views.get(obj_id, []):
                candidate_views.add(view_id)

        # If target objects have no views, use all views sorted by total visibility
        if not candidate_views:
            candidate_views = set(self.view_to_objects.keys())

        while len(selected) < min_count:
            best_view = None
            best_score = 0.0
            for view_id in candidate_views - set(selected):
                base_total = sum(
                    score for _, score in self.view_to_objects.get(view_id, [])
                )
                score = self._apply_pose_aware_gain(
                    base_gain=base_total,
                    view_id=view_id,
                    selected=selected,
                    frustum_overlap_threshold=frustum_overlap_threshold,
                    redundancy_penalty=redundancy_penalty,
                    dwell_weight=dwell_weight,
                    frustum_method=frustum_method,
                )
                if score > best_score:
                    best_score = score
                    best_view = view_id

            if best_view is None:
                break
            selected.append(best_view)

        return selected

    def _pad_keyframes_to_minimum_v14(
        self,
        selected: list[int],
        object_ids: list[int],
        min_count: int = 3,
    ) -> list[int]:
        """Preserve the original visibility-only padding behavior exactly."""
        if len(selected) >= min_count:
            return selected

        selected = list(selected)
        selected_set = set(selected)

        candidate_views: set[int] = set()
        for obj_id in object_ids:
            for view_id, _ in self.object_to_views.get(obj_id, []):
                candidate_views.add(view_id)

        if not candidate_views:
            candidate_views = set(self.view_to_objects.keys())

        unused = candidate_views - selected_set

        view_total_scores: list[tuple[int, float]] = []
        for view_id in unused:
            total = sum(score for _, score in self.view_to_objects.get(view_id, []))
            view_total_scores.append((view_id, total))
        view_total_scores.sort(key=lambda x: x[1], reverse=True)

        for view_id, _ in view_total_scores:
            if len(selected) >= min_count:
                break
            selected.append(view_id)

        return selected

    def _spatial_filter(
        self,
        candidates: list[SceneObject],
        anchor: SceneObject,
        relation: str,
    ) -> list[SceneObject]:
        """Filter candidates by spatial relation to anchor.

        Uses the SpatialRelationChecker for accurate relation checking.
        """
        if self._relation_checker is None:
            self._relation_checker = SpatialRelationChecker()

        filtered = []
        for obj in candidates:
            result = self._relation_checker.check(obj, anchor, relation)
            if result.satisfies:
                filtered.append((obj, result.score))

        # Sort by score and return objects
        if filtered:
            filtered.sort(key=lambda x: x[1], reverse=True)
            return [f[0] for f in filtered]

        # Fallback: return original candidates if none matched
        logger.warning(f"No candidates matched relation '{relation}', returning all")
        return candidates

    def _spatial_filter_multi_anchor(
        self,
        candidates: list[SceneObject],
        anchors: list[SceneObject],
        relation: str,
    ) -> list[SceneObject]:
        """Filter candidates by spatial relation to any of the anchors.

        Returns candidates that satisfy the relation with at least one anchor.
        """
        if self._relation_checker is None:
            self._relation_checker = SpatialRelationChecker()

        filtered = []
        for obj in candidates:
            best_score = 0.0
            satisfies_any = False

            for anchor in anchors:
                result = self._relation_checker.check(obj, anchor, relation)
                if result.satisfies:
                    satisfies_any = True
                    best_score = max(best_score, result.score)

            if satisfies_any:
                filtered.append((obj, best_score))

        if filtered:
            filtered.sort(key=lambda x: x[1], reverse=True)
            return [f[0] for f in filtered]

        return candidates

    def get_image(self, view_id: int) -> np.ndarray | None:
        """Load RGB image for a view."""
        if view_id >= len(self.image_paths):
            return None

        import cv2

        img = cv2.imread(str(self.image_paths[view_id]))
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None

    # ========== Hypothesis-Based Query Support (V3) ==========

    def _get_query_parser(self) -> QueryParser:
        """Get or create the query parser."""
        if self._query_parser is None:
            if self.llm_model is None:
                raise ValueError("llm_model is required for nested query parsing")
            self._query_parser = QueryParser(
                llm_model=self.llm_model,
                scene_categories=self.scene_categories,
                use_pool=self.use_pool,
            )
        return self._query_parser

    def _get_query_executor(self) -> QueryExecutor:
        """Get or create the query executor.

        Passes camera_poses to the executor so viewer-frame constraints can
        prefer the trajectory-based pose resolver over the geometric fallback.
        """
        if self._query_executor is None:
            if self._relation_checker is None:
                self._relation_checker = SpatialRelationChecker()

            self._query_executor = QueryExecutor(
                objects=self.objects,
                relation_checker=self._relation_checker,
                clip_features=self.object_features,
                clip_encoder=self._encode_text if HAS_CLIP else None,
                camera_poses=list(self.camera_poses) if self.camera_poses else None,
            )
        return self._query_executor

    def _generate_scene_images(self) -> list[str]:
        """
        Generate scene visualization images for multimodal query parsing.

        Uses OpenEQA ScanNet BEV (high-quality mesh-based rendering with
        frustum filtering). Raises on failure — no silent fallback.

        Returns:
            List of image paths (currently k=1)
        """
        from .bev_builder import OpenEQAScanNetBEVBuilder

        builder = OpenEQAScanNetBEVBuilder()
        _, bev_path = builder.build_from_scene(scene_path=self.scene_path)
        logger.info(f"[KeyframeSelector] ScanNet BEV ready: {bev_path}")
        return [str(bev_path)]

    def normalize_hypothesis_output(self, payload: Any) -> HypothesisOutputV1:
        """
        Normalize parser payload to HypothesisOutputV1.

        Supported input forms:
        1) HypothesisOutputV1
        2) GroundingQuery
        3) dict with `format_version=hypothesis_output_v1`
        4) legacy dict with `grounding_query`
        5) legacy GroundingQuery dict with `root`
        """
        if isinstance(payload, HypothesisOutputV1):
            output = payload
        elif isinstance(payload, GroundingQuery):
            output = HypothesisOutputV1.from_direct_query(payload)
        elif isinstance(payload, dict):
            if payload.get("format_version") == "hypothesis_output_v1":
                output = HypothesisOutputV1.model_validate(payload)
            elif "grounding_query" in payload:
                grounding_query = self.to_grounding_query(payload)
                output = HypothesisOutputV1(
                    parse_mode=ParseMode.SINGLE,
                    hypotheses=[
                        QueryHypothesis(
                            kind=payload.get("kind", HypothesisKind.DIRECT),
                            rank=1,
                            grounding_query=grounding_query,
                            lexical_hints=list(payload.get("lexical_hints", [])),
                        )
                    ],
                )
            elif "root" in payload:
                grounding_query = GroundingQuery.model_validate(payload)
                output = HypothesisOutputV1.from_direct_query(grounding_query)
            else:
                raise ValueError(
                    "Unsupported payload dict format for hypothesis output"
                )
        else:
            raise TypeError(
                f"Unsupported payload type for hypothesis normalization: {type(payload)}"
            )

        ordered = output.ordered_hypotheses()
        if [h.rank for h in ordered] != [h.rank for h in output.hypotheses]:
            output = HypothesisOutputV1(
                parse_mode=output.parse_mode, hypotheses=ordered
            )
        return output

    def to_grounding_query(self, hypothesis_payload: Any) -> GroundingQuery:
        """
        Convert one hypothesis payload to GroundingQuery with strict validation.

        This method never silently skips malformed payload.
        """
        if isinstance(hypothesis_payload, GroundingQuery):
            return hypothesis_payload
        if isinstance(hypothesis_payload, QueryHypothesis):
            return hypothesis_payload.grounding_query

        if isinstance(hypothesis_payload, dict):
            raw = hypothesis_payload.get("grounding_query", hypothesis_payload)
            if not isinstance(raw, dict):
                raise ValueError("grounding_query must be a dict payload")
            return GroundingQuery.model_validate(raw)

        raise TypeError(
            f"Unsupported hypothesis payload type for GroundingQuery conversion: {type(hypothesis_payload)}"
        )

    def _iter_query_nodes(self, root: QueryNode):
        """Depth-first traversal over query nodes."""
        stack = [root]
        while stack:
            node = stack.pop()
            yield node
            for constraint in node.spatial_constraints:
                stack.extend(constraint.anchors)
            if node.select_constraint and node.select_constraint.reference:
                stack.append(node.select_constraint.reference)

    def _iter_viewpoint_anchor_nodes(
        self, grounding_query: GroundingQuery
    ) -> Iterable[QueryNode]:
        """Yield every QueryNode embedded inside a ViewpointContext."""
        for vc in grounding_query.viewpoint_contexts:
            for anchor in (vc.facing_anchor, vc.origin_anchor, vc.subject_anchor):
                if anchor is None:
                    continue
                yield from self._iter_query_nodes(anchor)

    def _sanitize_grounding_query_categories(
        self, grounding_query: GroundingQuery
    ) -> GroundingQuery:
        """
        Ensure all executable categories are in scene categories or UNKNOW.

        Also sanitizes viewpoint context anchor categories using the same
        rule, so subsequent validate_categories() does not blow up on a
        category that the LLM emitted only inside a viewpoint context.
        """
        scene_set = set(self.scene_categories)
        sanitized = grounding_query.model_copy(deep=True)

        all_nodes = list(self._iter_query_nodes(sanitized.root)) + list(
            self._iter_viewpoint_anchor_nodes(sanitized)
        )
        for node in all_nodes:
            cleaned = []
            seen = set()
            for cat in node.categories:
                if cat in scene_set or cat == "UNKNOW":
                    if cat not in seen:
                        cleaned.append(cat)
                        seen.add(cat)
            node.categories = cleaned if cleaned else ["UNKNOW"]

        # Mark root as open_ended when target is UNKNOW with spatial constraints,
        # but NOT for comparative/attributive queries about the anchor itself
        # (e.g., "Which side of the closet is emptier?" should keep open_ended=False)
        root = sanitized.root
        if (
            root.categories == ["UNKNOW"]
            and root.spatial_constraints
            and any(
                "UNKNOW" not in anchor.categories
                for sc in root.spatial_constraints
                for anchor in sc.anchors
            )
        ):
            # Skip open_ended for comparative relations that ask about anchor properties
            comparative_relations = {
                "side_of",
                "part_of",
                "section_of",
                "half_of",
                "top_of",
                "bottom_of",
                "left_of",
                "right_of",
            }
            is_comparative = any(
                sc.relation.lower().replace(" ", "_") in comparative_relations
                for sc in root.spatial_constraints
            )
            if not is_comparative:
                root.open_ended = True

        return sanitized

    def validate_categories_in_scene(self, grounding_query: GroundingQuery) -> None:
        """Validate categories in one GroundingQuery against scene categories."""
        scene_set = set(self.scene_categories)
        for cat in grounding_query.get_all_categories():
            if cat != "UNKNOW" and cat not in scene_set:
                raise ValueError(
                    f"Category '{cat}' is not in scene categories and is not UNKNOW"
                )

    def validate_no_mask_leak(
        self,
        grounding_query: GroundingQuery,
        hidden_categories: Iterable[str] | None,
    ) -> None:
        """Validate one GroundingQuery does not leak hidden categories."""
        hidden_set = set(hidden_categories or [])
        if not hidden_set:
            return

        for cat in grounding_query.get_all_categories():
            if cat in hidden_set:
                raise ValueError(f"Masked category leak detected: '{cat}'")

    def parse_query_hypotheses(
        self,
        query: str,
        max_hypotheses: int = 3,
        use_visual_context: bool = True,
        apply_viewpoint_normalize: bool = False,
    ) -> HypothesisOutputV1:
        """
        Parse query into the unified HypothesisOutputV1 structure.

        The LLM directly outputs HypothesisOutputV1 with all hypotheses.
        This method always sanitizes categories. Viewpoint-aware normalize
        is OPT-IN to preserve legacy production behavior.

        Args:
            query: Natural language query string
            max_hypotheses: Maximum hypotheses (ignored - LLM decides)
            use_visual_context: If True (default), generate BEV image for multimodal parsing
            apply_viewpoint_normalize: When True, apply the Phase-1 policy floor
                (directional-without-context -> ambiguous/rank_only) plus the
                ungrounded-viewer demotion. Default False so existing callers
                see legacy behavior (no policy downgrade) unless they opt in.

        Returns:
            HypothesisOutputV1 with hypotheses ready for execution
        """
        parser = self._get_query_parser()

        # Generate scene images if visual context requested
        scene_images = None
        if use_visual_context:
            scene_images = self._generate_scene_images()

        # LLM directly outputs HypothesisOutputV1
        output = parser.parse(query, scene_images=scene_images)

        # Sanitize categories in each hypothesis
        sanitized_hypotheses = []
        for hypo in output.hypotheses:
            sanitized_gq = self._sanitize_grounding_query_categories(
                hypo.grounding_query
            )
            if apply_viewpoint_normalize:
                sanitized_gq = self._normalize_viewpoint_policies(sanitized_gq)
            sanitized_hypo = QueryHypothesis(
                kind=hypo.kind,
                rank=hypo.rank,
                grounding_query=sanitized_gq,
                lexical_hints=hypo.lexical_hints,
            )
            sanitized_hypotheses.append(sanitized_hypo)

        sanitized_output = HypothesisOutputV1(
            parse_mode=output.parse_mode,
            hypotheses=sanitized_hypotheses,
        )
        sanitized_output.validate_categories(self.scene_categories)
        return sanitized_output

    def _force_legacy_world_hard(
        self, output: HypothesisOutputV1
    ) -> HypothesisOutputV1:
        """Restore strict legacy semantics on every constraint.

        Used when ``viewpoint_aware=False`` to make the executor see exactly
        what it would have seen pre-patch, regardless of what the LLM emitted.
        For every spatial/select constraint:
        - reference_frame -> WORLD
        - viewpoint_context_id -> None
        - execution_policy -> HARD
        Also clears ``grounding_query.viewpoint_contexts`` so downstream
        consumers don't think viewpoint info is active.
        """
        new_hypos: list[QueryHypothesis] = []
        for hypo in output.hypotheses:
            gq = hypo.grounding_query.model_copy(deep=True)
            for _node, constraint in gq.iter_constraints():
                constraint.reference_frame = ReferenceFrame.WORLD
                constraint.viewpoint_context_id = None
                constraint.execution_policy = ExecutionPolicy.HARD
            gq.viewpoint_contexts = []
            new_hypos.append(
                QueryHypothesis(
                    kind=hypo.kind,
                    rank=hypo.rank,
                    grounding_query=gq,
                    lexical_hints=hypo.lexical_hints,
                )
            )
        return HypothesisOutputV1(parse_mode=output.parse_mode, hypotheses=new_hypos)

    def _guard_viewpoint_traj_available(self, output: HypothesisOutputV1) -> None:
        """Strict-no-fallback: if any viewer-frame constraint is requested
        but no usable camera trajectory could be loaded, RAISE.

        Called only when ``viewpoint_aware=True``. The failure surface is
        opt-in: legacy callers (``viewpoint_aware=False``) never reach this
        guard. When raised, the message lists every path that was probed so
        the operator can fix the data layout.
        """
        if self.camera_poses:
            return
        has_viewer_constraint = False
        for hypo in output.hypotheses:
            for _node, constraint in hypo.grounding_query.iter_constraints():
                if constraint.reference_frame == ReferenceFrame.VIEWER:
                    has_viewer_constraint = True
                    break
            if has_viewer_constraint:
                break
        if not has_viewer_constraint:
            return

        tried = self._describe_trajectory_search_paths()
        raise RuntimeError(
            "viewpoint_aware=True requires a usable camera trajectory, but "
            "none of the probed paths returned any 4x4 poses.\n"
            f"  scene_path: {self.scene_path}\n"
            f"  paths tried: {tried}\n"
            "Provide a traj.txt (16-per-line or 4-rows-per-pose) or a pack "
            "camera_trajectory.json with full 4x4 poses, or disable "
            "viewpoint_aware to use the strict no-drift legacy path "
            "(which collapses every constraint to world/hard before execution "
            "via _force_legacy_world_hard())."
        )

    def _rank_target_objects_by_score(
        self, result: ExecutionResult
    ) -> list[SceneObject]:
        """Sort matched_objects by multiplicative score, with soft_match tie-break.

        Under HARD policy every surviving object has score 1.0, so this is a
        stable identity transform (matches legacy behavior). Under SOFT the
        multiplicative score discriminates; under RANK_ONLY the multiplicative
        score is uniformly 1.0 but ``metadata["soft_match_scores"]`` carries
        per-constraint satisfaction that we sum as a tie-breaker.
        """
        objects = list(result.matched_objects)
        if not objects:
            return objects

        scores = dict(result.scores or {})
        soft_map: dict[int, dict[str, float]] = result.metadata.get(
            "soft_match_scores", {}
        )

        def soft_total(obj_id: int) -> float:
            entries = soft_map.get(obj_id, {})
            return sum(entries.values()) if entries else 0.0

        def sort_key(obj: SceneObject) -> tuple[float, float]:
            return (
                -scores.get(obj.obj_id, 0.0),
                -soft_total(obj.obj_id),
            )

        return sorted(objects, key=sort_key)

    def _normalize_viewpoint_policies(
        self, grounding_query: GroundingQuery
    ) -> GroundingQuery:
        """Post-parse viewpoint-aware normalization.

        Two passes that together implement the Phase-1 policy floor:

        1. Directional relation (left_of / right_of / in_front_of / behind)
           with default world frame and no viewpoint context: demote to
           ``frame=ambiguous`` + ``policy=rank_only``. This stops these
           relations from filtering candidates to empty - matches the
           empirical NR3D failure mode where 116 / 151 no_evidence rows
           already have the GT target in the category-match set.

        2. Viewer / object_local constraint whose viewpoint context anchor
           sanitizes to ["UNKNOW"] (i.e. no scene category resolves): the
           context is ungrounded, so demote the constraint to
           ``frame=ambiguous`` + ``policy=rank_only`` rather than asking the
           executor to compute a viewer pose from nothing.
        """
        gq = grounding_query.model_copy(deep=True)

        ungrounded_context_ids: set[str] = set()
        for vc in gq.viewpoint_contexts:
            anchors_have_real_categories = False
            for anchor in (vc.facing_anchor, vc.origin_anchor, vc.subject_anchor):
                if anchor is None:
                    continue
                for node in self._iter_query_nodes(anchor):
                    if any(cat != "UNKNOW" for cat in node.categories):
                        anchors_have_real_categories = True
                        break
                if anchors_have_real_categories:
                    break
            if not anchors_have_real_categories:
                ungrounded_context_ids.add(vc.id)

        for _node, constraint in gq.iter_constraints():
            if isinstance(constraint, SpatialConstraint):
                relation_norm = constraint.relation.lower().replace(" ", "_")
                is_directional = relation_norm in DIRECTIONAL_RELATIONS
            elif isinstance(constraint, SelectConstraint):
                metric_norm = constraint.metric.lower().replace(" ", "_")
                is_directional = metric_norm in {
                    "x_position",
                    "x",
                    "y_position",
                    "y",
                }
            else:
                is_directional = False

            # Pass 1: world-frame directional without context -> ambiguous/rank_only
            if (
                is_directional
                and constraint.reference_frame == ReferenceFrame.WORLD
                and constraint.viewpoint_context_id is None
                and constraint.execution_policy == ExecutionPolicy.HARD
            ):
                constraint.reference_frame = ReferenceFrame.AMBIGUOUS
                constraint.execution_policy = ExecutionPolicy.RANK_ONLY
                continue

            # Pass 2: viewer/object_local with ungrounded context -> ambiguous/rank_only
            if (
                constraint.reference_frame
                in (ReferenceFrame.VIEWER, ReferenceFrame.OBJECT_LOCAL)
                and constraint.viewpoint_context_id in ungrounded_context_ids
            ):
                constraint.reference_frame = ReferenceFrame.AMBIGUOUS
                constraint.viewpoint_context_id = None
                constraint.execution_policy = ExecutionPolicy.RANK_ONLY

        return gq

    def _has_unknown_anchors(self, grounding_query: GroundingQuery) -> bool:
        """Check if any anchor or reference in the query has UNKNOW category."""
        for node in self._iter_query_nodes(grounding_query.root):
            # Check spatial constraint anchors
            for sc in node.spatial_constraints:
                for anchor in sc.anchors:
                    if "UNKNOW" in anchor.categories:
                        return True
            # Check select constraint reference
            if node.select_constraint and node.select_constraint.reference:
                if "UNKNOW" in node.select_constraint.reference.categories:
                    return True
        return False

    def execute_query(
        self,
        grounding_query: GroundingQuery,
        mode: ExecutionMode | str = ExecutionMode.STRICT,
    ) -> ExecutionResult:
        """Execute a parsed grounding query.

        Args:
            grounding_query: Parsed GroundingQuery object
            mode: Strict or recall-biased execution.

        Returns:
            ExecutionResult with matched objects
        """
        executor = self._get_query_executor()
        return executor.execute(grounding_query, mode=mode)

    # TODO(v9.2): No remaining consumers after select_by_hypothesis was deleted in v9.1.
    # Verify no external benchmark scripts depend on this before removing.
    def execute_hypotheses(
        self,
        hypothesis_output: Any,
        hidden_categories: Iterable[str] | None = None,
        mode: ExecutionMode | str = ExecutionMode.STRICT,
    ) -> tuple[str, QueryHypothesis | None, ExecutionResult]:
        """
        Execute hypotheses by rank and return first non-empty result.

        Special handling:
        - If a hypothesis has UNKNOW anchors/references, skip it even if executor
          returns results (because those results don't satisfy the spatial constraint).
        """
        if not isinstance(mode, ExecutionMode):
            mode = ExecutionMode(mode)

        normalized = self.normalize_hypothesis_output(hypothesis_output)
        normalized.validate_categories(self.scene_categories)
        attempted_traces: list[dict[str, Any]] = []

        for hypothesis in normalized.ordered_hypotheses():
            grounding_query = self.to_grounding_query(hypothesis)
            self.validate_categories_in_scene(grounding_query)
            self.validate_no_mask_leak(grounding_query, hidden_categories)

            # Skip hypothesis if it has UNKNOW anchors - spatial constraint can't be satisfied
            if self._has_unknown_anchors(grounding_query):
                logger.debug(
                    f"[execute_hypotheses] Skipping hypothesis {hypothesis.kind.value} "
                    f"with UNKNOW anchors"
                )
                attempted_traces.append(
                    {
                        "kind": hypothesis.kind.value,
                        "rank": hypothesis.rank,
                        "status": "skipped_unknown_anchor",
                        "trace": [],
                    }
                )
                continue

            if mode == ExecutionMode.STRICT:
                result = self.execute_query(grounding_query)
            else:
                result = self.execute_query(grounding_query, mode=mode)
            if result.is_empty:
                attempted_traces.append(
                    {
                        "kind": hypothesis.kind.value,
                        "rank": hypothesis.rank,
                        "status": "empty",
                        "trace": result.metadata.get("execution_trace", []),
                    }
                )
                continue

            if attempted_traces:
                result.metadata = dict(result.metadata)
                result.metadata["execution_trace_attempts"] = attempted_traces
            if hypothesis.kind == HypothesisKind.DIRECT:
                return "direct_grounded", hypothesis, result
            if hypothesis.kind == HypothesisKind.PROXY:
                return "proxy_grounded", hypothesis, result
            return "context_only", hypothesis, result

        return (
            "no_evidence",
            None,
            ExecutionResult(
                node_id="none",
                matched_objects=[],
                metadata={"execution_trace_attempts": attempted_traces},
            ),
        )

    def map_view_to_frame(self, view_id: int) -> int:
        """Map sampled view index to original frame index."""
        return view_id * self.stride

    def _resolve_keyframe_path(self, view_id: int) -> tuple[Path | None, int]:
        """
        Resolve image path from view index with stride-aware fallback.

        Returns:
            Tuple[path, resolved_view_id]
        """
        search_order = [view_id, view_id - 1, view_id + 1, view_id - 2, view_id + 2]
        for cand_view in search_order:
            if cand_view < 0:
                continue

            cand_frame = self.map_view_to_frame(cand_view)
            expected = self.scene_path / "results" / f"frame{cand_frame:06d}.jpg"
            if expected.exists():
                return expected, cand_view

            if 0 <= cand_view < len(self.image_paths):
                fallback = self.image_paths[cand_view]
                if fallback.exists():
                    return fallback, cand_view

        return None, view_id

    def select_keyframes_v2(
        self,
        query: str,
        k: int = 3,
        strategy: str = "joint_coverage",
        hidden_categories: Iterable[str] | None = None,
        use_visual_context: bool = True,
        pose_aware: bool = False,
        frustum_method: str = "l1",
        viewpoint_aware: bool = False,
    ) -> KeyframeResult:
        """
        Select keyframes from the new structured output `HypothesisOutputV1`.

        Execution order:
        1) Parse query into hypotheses (single or multi)
        2) Execute hypotheses by rank
        3) Select keyframes from first successful hypothesis

        Args:
            use_visual_context: If True, generate BEV image for multimodal
                query parsing. Set False when scene mesh is unavailable.
            viewpoint_aware: Phase-3 feature flag. When True, the parser
                is permitted to emit ViewpointContext, the Phase-1 policy
                floor is applied (directional-without-context becomes
                rank_only via ``_normalize_viewpoint_policies``), and the
                executor honours ``reference_frame=viewer`` with
                trajectory-mode viewer-frame geometry.
                When False (default), the path is strict no-drift legacy:
                ``_force_legacy_world_hard()`` rewrites EVERY constraint
                back to ``reference_frame=WORLD`` / ``execution_policy=HARD``
                and clears ``grounding_query.viewpoint_contexts`` before
                the executor sees it. No Phase-1 policy floor is applied
                and no viewer-frame geometry is consulted - production
                ``select_by_text`` behaves exactly as it did pre-patch.
        """
        self.pose_aware_enabled = pose_aware
        logger.info(
            f"[V3] Selecting {k} keyframes for: '{query}' "
            f"(viewpoint_aware={viewpoint_aware})"
        )

        # Step 1: Parse to new unified structure. The viewpoint policy floor
        # is OFF by default so legacy callers see strict no-drift behavior.
        hypothesis_output = self.parse_query_hypotheses(
            query,
            max_hypotheses=3,
            use_visual_context=use_visual_context,
            apply_viewpoint_normalize=viewpoint_aware,
        )

        if not viewpoint_aware:
            # Strict no-drift legacy path: force every constraint back to
            # WORLD / HARD before the executor sees it, so the existence of
            # viewpoint emission in the parser cannot change production
            # select_by_text behavior.
            hypothesis_output = self._force_legacy_world_hard(hypothesis_output)
        else:
            self._guard_viewpoint_traj_available(hypothesis_output)
        logger.info(
            f"[V3] Parsed format={hypothesis_output.format_version}, "
            f"mode={hypothesis_output.parse_mode.value}, "
            f"hypotheses={len(hypothesis_output.hypotheses)}"
        )

        # Step 2: Execute hypotheses by rank
        execution_mode = (
            ExecutionMode.RECALL if viewpoint_aware else ExecutionMode.STRICT
        )
        status, selected_hypothesis, result = self.execute_hypotheses(
            hypothesis_output=hypothesis_output,
            hidden_categories=hidden_categories,
            mode=execution_mode,
        )

        if status == "no_evidence" or selected_hypothesis is None or result.is_empty:
            logger.warning("[V3] No executable hypothesis produced evidence")
            first = hypothesis_output.ordered_hypotheses()[0]
            return KeyframeResult(
                query=query,
                target_term=first.grounding_query.root.category,
                anchor_term=None,
                keyframe_indices=[],
                keyframe_paths=[],
                target_objects=[],
                anchor_objects=[],
                metadata={
                    "status": status,
                    "error": "No matching objects",
                    "hypothesis_output": hypothesis_output.model_dump(),
                    "execution_mode": execution_mode.value,
                    "execution_trace": result.metadata.get("execution_trace", []),
                    "execution_trace_attempts": result.metadata.get(
                        "execution_trace_attempts", []
                    ),
                    "version": "v3",
                },
            )

        target_objects = self._rank_target_objects_by_score(result)
        selected_query = selected_hypothesis.grounding_query
        logger.info(
            f"[V3] Selected hypothesis kind={selected_hypothesis.kind.value}, "
            f"matched={len(target_objects)}"
        )

        # Step 3: Collect anchor objects for joint coverage
        anchor_objects: list[SceneObject] = []
        between_anchors: list[SceneObject] = []
        for constraint in selected_query.root.spatial_constraints:
            for anchor_node in constraint.anchors:
                anchor_result = self._get_query_executor()._execute_node(anchor_node)
                anchor_objects.extend(anchor_result.matched_objects)
            # Track "between" anchor pairs for midpoint retrieval
            if constraint.relation.lower() == "between" and len(anchor_objects) >= 2:
                between_anchors = anchor_objects[-2:]

        # Step 3b: For "between" queries, find objects near the spatial midpoint
        midpoint_object_ids: list[int] = []
        if between_anchors and len(between_anchors) >= 2:
            c1 = getattr(between_anchors[0], "centroid", None)
            c2 = getattr(between_anchors[1], "centroid", None)
            if c1 is not None and c2 is not None:
                import numpy as np

                midpoint = (np.asarray(c1) + np.asarray(c2)) / 2.0
                # Find objects closest to midpoint for view selection
                obj_dists = []
                for obj in self.objects:
                    oc = getattr(obj, "centroid", None)
                    if oc is not None:
                        dist = float(np.linalg.norm(np.asarray(oc) - midpoint))
                        obj_dists.append((obj.obj_id, dist))
                obj_dists.sort(key=lambda x: x[1])
                midpoint_object_ids = [oid for oid, _ in obj_dists[:5]]
                logger.info(
                    f"[V3] 'between' query: midpoint objects {midpoint_object_ids}"
                )

        # Step 4: Select keyframes
        all_object_ids = [obj.obj_id for obj in target_objects[:5]]
        if anchor_objects:
            all_object_ids.extend([obj.obj_id for obj in anchor_objects[:3]])
        if midpoint_object_ids:
            all_object_ids.extend(midpoint_object_ids)

        if strategy == "joint_coverage":
            keyframe_indices = self.get_joint_coverage_views(
                all_object_ids,
                max_views=k,
                pose_aware=pose_aware,
                frustum_method=frustum_method,
            )
        else:
            keyframe_indices = self.get_joint_coverage_views(
                [obj.obj_id for obj in target_objects[:5]],
                max_views=k,
                pose_aware=pose_aware,
                frustum_method=frustum_method,
            )

        # Ensure minimum keyframe count (pad with high-visibility views)
        keyframe_indices = self._pad_keyframes_to_minimum(
            keyframe_indices,
            all_object_ids,
            min_count=min(k, 3),
            pose_aware=pose_aware,
            frustum_method=frustum_method,
        )

        keyframe_paths = []
        frame_mappings = []
        for view_id in keyframe_indices:
            requested_frame_id = self.map_view_to_frame(view_id)
            path, resolved_view_id = self._resolve_keyframe_path(view_id)
            resolved_frame_id = self.map_view_to_frame(resolved_view_id)
            if path is not None:
                keyframe_paths.append(path)
            frame_mappings.append(
                {
                    "requested_view_id": view_id,
                    "requested_frame_id": requested_frame_id,
                    "resolved_view_id": resolved_view_id,
                    "resolved_frame_id": resolved_frame_id,
                    "path": str(path) if path is not None else None,
                }
            )

        # Extract anchor term
        anchor_term = None
        if selected_query.root.spatial_constraints:
            anchors = selected_query.root.spatial_constraints[0].anchors
            if anchors:
                anchor_term = anchors[0].category

        return KeyframeResult(
            query=query,
            target_term=selected_query.root.category,
            anchor_term=anchor_term,
            keyframe_indices=keyframe_indices,
            keyframe_paths=keyframe_paths,
            target_objects=target_objects,
            anchor_objects=anchor_objects,
            metadata={
                "status": status,
                "selected_hypothesis_kind": selected_hypothesis.kind.value,
                "selected_hypothesis_rank": selected_hypothesis.rank,
                "strategy": strategy,
                "all_object_ids": all_object_ids,
                "frame_mappings": frame_mappings,
                "hypothesis_output": hypothesis_output.model_dump(),
                "execution_mode": execution_mode.value,
                "execution_trace": result.metadata.get("execution_trace", []),
                "execution_trace_attempts": result.metadata.get(
                    "execution_trace_attempts", []
                ),
                "version": "v3",
            },
        )


# Convenience function
def select_keyframes(
    scene_path: str,
    query: str,
    k: int = 3,
    **kwargs,
) -> KeyframeResult:
    """Convenience function to select keyframes for a query.

    Args:
        scene_path: Path to scene directory
        query: Natural language query
        k: Number of keyframes
        **kwargs: Additional arguments for KeyframeSelector

    Returns:
        KeyframeResult
    """
    selector = KeyframeSelector.from_scene_path(scene_path, **kwargs)
    return selector.select_keyframes_v2(query, k=k)

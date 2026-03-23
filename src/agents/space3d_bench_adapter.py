"""Space3D-Bench adapter for Stage 2 agent evaluation.

Space3D-Bench provides:
- 1000 spatial QA questions across 13 Replica scenes
- detections.json with object info (class, center, size, rotation)
- navmesh.txt for navigation queries
- img/ with reference images for subjective question evaluation

This adapter enables running Stage 2 agent on Space3D-Bench questions
using the benchmark-provided object detections as scene context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import (
    KeyframeEvidence,
    Stage1HypothesisSummary,
    Stage2EvidenceBundle,
    Stage2TaskSpec,
)


@dataclass
class Space3DBenchSample:
    """A single Space3D-Bench question."""

    scene_id: str
    question_id: str
    question: str
    ground_truth: dict[str, Any]  # answer + prompt
    image_paths: list[Path] = field(default_factory=list)  # Reference images if any


@dataclass
class Space3DBenchScene:
    """Scene data from Space3D-Bench."""

    scene_id: str
    questions: dict[str, str]  # question_id -> question text
    ground_truth: dict[str, dict[str, Any]]  # question_id -> {answer, prompt}
    detections: dict[str, Any]  # Object detections
    navmesh_path: Path | None = None
    image_dir: Path | None = None

    def get_sample(self, question_id: str) -> Space3DBenchSample:
        """Get a single sample by question ID."""
        question = self.questions.get(question_id, "")
        gt = self.ground_truth.get(question_id, {})

        # Check if this question has associated images
        image_paths = []
        if self.image_dir and self.image_dir.exists():
            # Images are named like q21.png, q22.png
            img_path = self.image_dir / f"q{question_id}.png"
            if img_path.exists():
                image_paths.append(img_path)

        return Space3DBenchSample(
            scene_id=self.scene_id,
            question_id=question_id,
            question=question,
            ground_truth=gt,
            image_paths=image_paths,
        )

    def iter_samples(self):
        """Iterate over all samples in the scene."""
        for qid in self.questions:
            yield self.get_sample(qid)


class Space3DBenchLoader:
    """Loader for Space3D-Bench dataset."""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self._scenes: dict[str, Space3DBenchScene] = {}

    def list_scenes(self) -> list[str]:
        """List available scenes."""
        scenes = []
        for d in self.data_root.iterdir():
            if d.is_dir() and (d / "questions.json").exists():
                scenes.append(d.name)
        return sorted(scenes)

    def load_scene(self, scene_id: str) -> Space3DBenchScene:
        """Load a scene's data."""
        if scene_id in self._scenes:
            return self._scenes[scene_id]

        scene_dir = self.data_root / scene_id

        # Load questions
        questions_path = scene_dir / "questions.json"
        with open(questions_path) as f:
            questions = json.load(f)

        # Load ground truth
        gt_path = scene_dir / "ground_truth.json"
        with open(gt_path) as f:
            ground_truth = json.load(f)

        # Load detections
        detections_path = scene_dir / "misc" / "detections.json"
        detections = {}
        if detections_path.exists():
            with open(detections_path) as f:
                detections = json.load(f)

        # Navmesh path
        navmesh_path = scene_dir / "misc" / "navmesh.txt"
        if not navmesh_path.exists():
            navmesh_path = None

        # Image directory
        image_dir = scene_dir / "img"
        if not image_dir.exists():
            image_dir = None

        scene = Space3DBenchScene(
            scene_id=scene_id,
            questions=questions,
            ground_truth=ground_truth,
            detections=detections,
            navmesh_path=navmesh_path,
            image_dir=image_dir,
        )

        self._scenes[scene_id] = scene
        return scene

    def get_sample(self, scene_id: str, question_id: str) -> Space3DBenchSample:
        """Get a specific sample."""
        scene = self.load_scene(scene_id)
        return scene.get_sample(question_id)


def build_scene_summary_from_detections(detections: dict[str, Any]) -> str:
    """Build a scene summary from Space3D-Bench detections."""
    if not detections:
        return "No object detection data available."

    # Count objects by class
    class_counts: dict[str, int] = {}
    room_info = detections.get("room", {})
    objects = room_info.get("objects", {})

    for obj_id, obj_data in objects.items():
        class_name = obj_data.get("class_name", "unknown")
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    # Build summary
    lines = [f"Scene contains {len(objects)} detected objects:"]
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  - {cls}: {count}")

    return "\n".join(lines)


def build_object_context_from_detections(detections: dict[str, Any]) -> dict[str, str]:
    """Build object context dict from Space3D-Bench detections.

    Returns Dict[str, str] where key is object_id and value is a text description
    to match the Stage2EvidenceBundle schema.
    """
    if not detections:
        return {}

    room_info = detections.get("room", {})
    objects = room_info.get("objects", {})

    context: dict[str, str] = {}
    for obj_id, obj_data in objects.items():
        class_name = obj_data.get("class_name", "unknown")
        center = obj_data.get("center", [0, 0, 0])
        sizes = obj_data.get("sizes", [0, 0, 0])

        # Format as descriptive string for the agent
        pos_str = f"({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})"
        size_str = f"{sizes[0]:.2f}x{sizes[1]:.2f}x{sizes[2]:.2f}"
        context[obj_id] = f"{class_name} at {pos_str}, size {size_str}m"

    return context


def build_evidence_bundle_for_space3d(
    sample: Space3DBenchSample,
    detections: dict[str, Any],
    extra_image_paths: list[Path] | None = None,
) -> Stage2EvidenceBundle:
    """Build Stage2EvidenceBundle for a Space3D-Bench sample.

    Since Space3D-Bench doesn't provide per-question RGB frames,
    we use the reference images when available, or create a bundle
    with just the detection context.
    """
    keyframes = []

    # Use reference images if available
    all_images = list(sample.image_paths)
    if extra_image_paths:
        all_images.extend(extra_image_paths)

    for idx, img_path in enumerate(all_images):
        keyframes.append(
            KeyframeEvidence(
                keyframe_idx=idx,
                image_path=str(img_path),
                view_id=idx,
                frame_id=idx,
                score=1.0,
                note="space3d_bench_reference",
            )
        )

    # Build hypothesis
    hypothesis = Stage1HypothesisSummary(
        status="space3d_bench",
        hypothesis_kind="spatial_qa",
        hypothesis_rank=0,
        parse_mode="space3d_bench",
        raw_query=sample.question,
        target_categories=[],
        anchor_categories=[],
        metadata={
            "benchmark": "space3d_bench",
            "scene_id": sample.scene_id,
            "question_id": sample.question_id,
        },
    )

    # Build scene summary and object context
    scene_summary = build_scene_summary_from_detections(detections)
    object_context = build_object_context_from_detections(detections)

    return Stage2EvidenceBundle(
        scene_id=sample.scene_id,
        stage1_query=sample.question,
        keyframes=keyframes,
        bev_image_path=None,
        scene_summary=scene_summary,
        object_context=object_context,
        hypothesis=hypothesis,
        extra_metadata={
            "benchmark": "space3d_bench",
            "question_id": sample.question_id,
            "has_reference_images": len(keyframes) > 0,
            "num_objects": len(object_context),
        },
    )


def build_task_spec_for_space3d(
    sample: Space3DBenchSample,
    plan_mode: str = "off",
    max_reasoning_turns: int = 3,
) -> Stage2TaskSpec:
    """Build Stage2TaskSpec for a Space3D-Bench sample."""
    # Determine task type based on question content
    question_lower = sample.question.lower()

    if "3d position" in question_lower or "distance" in question_lower:
        task_type = "visual_grounding"
    elif "how many" in question_lower or "are there" in question_lower:
        task_type = "qa"
    elif "navigate" in question_lower or "walk" in question_lower:
        task_type = "nav_plan"
    else:
        task_type = "qa"

    return Stage2TaskSpec(
        task_type=task_type,
        user_query=sample.question,
        plan_mode=plan_mode,
        max_reasoning_turns=max_reasoning_turns,
    )


@dataclass
class Space3DBenchAdapter:
    """Adapter for running Stage 2 agent on Space3D-Bench."""

    loader: Space3DBenchLoader
    plan_mode: str = "off"
    max_reasoning_turns: int = 3

    @classmethod
    def from_data_root(cls, data_root: Path, **kwargs) -> Space3DBenchAdapter:
        """Create adapter from data root path."""
        loader = Space3DBenchLoader(data_root)
        return cls(loader=loader, **kwargs)

    def prepare_inputs(
        self,
        scene_id: str,
        question_id: str,
        extra_image_paths: list[Path] | None = None,
    ) -> tuple[Stage2TaskSpec, Stage2EvidenceBundle]:
        """Prepare Stage 2 inputs for a specific question."""
        scene = self.loader.load_scene(scene_id)
        sample = scene.get_sample(question_id)

        task = build_task_spec_for_space3d(
            sample,
            plan_mode=self.plan_mode,
            max_reasoning_turns=self.max_reasoning_turns,
        )

        bundle = build_evidence_bundle_for_space3d(
            sample,
            scene.detections,
            extra_image_paths,
        )

        return task, bundle

    def get_ground_truth(self, scene_id: str, question_id: str) -> dict[str, Any]:
        """Get ground truth for a question."""
        scene = self.loader.load_scene(scene_id)
        return scene.ground_truth.get(question_id, {})

    def iter_scene_samples(self, scene_id: str):
        """Iterate over all samples in a scene."""
        scene = self.loader.load_scene(scene_id)
        for sample in scene.iter_samples():
            yield sample


def evaluate_answer(
    prediction: str,
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    """Simple evaluation of predicted answer against ground truth.

    This is a basic string-matching evaluation. For proper evaluation,
    use the Space3D-Bench official evaluation scripts with VLM-based assessment.
    """
    gt_answer = ground_truth.get("answer", "")
    prompt = ground_truth.get("prompt", "")

    # Handle image-based answers (subjective questions)
    if isinstance(gt_answer, dict) and "image_path" in gt_answer:
        return {
            "type": "subjective",
            "requires_vlm_eval": True,
            "example_answer": gt_answer.get("example_answer", ""),
            "image_path": gt_answer.get("image_path"),
        }

    # Simple exact match for numerical answers
    result = {
        "type": "objective",
        "ground_truth": gt_answer,
        "prediction": prediction,
        "prompt": prompt,
    }

    # Try to extract numbers for comparison
    pred_lower = prediction.lower()
    gt_lower = str(gt_answer).lower()

    if "number of objects:" in gt_lower:
        try:
            gt_num = int(gt_lower.split(":")[-1].strip())
            # Try to find number in prediction
            import re

            numbers = re.findall(r"\d+", pred_lower)
            if numbers:
                pred_num = int(numbers[0])
                result["match"] = pred_num == gt_num
            else:
                result["match"] = False
        except:
            result["match"] = gt_lower in pred_lower
    elif "yes" in gt_lower or "no" in gt_lower:
        gt_bool = "yes" in gt_lower
        pred_bool = "yes" in pred_lower
        result["match"] = gt_bool == pred_bool
    else:
        # Fallback to substring match
        result["match"] = gt_lower in pred_lower or pred_lower in gt_lower

    return result

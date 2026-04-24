from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .detector_adapter import load_detector_proposals_json, model_blocked_record
from .models import DetectorInputRecord, ProposalRecord


def render_detector_command(
    *,
    command_template: str,
    input_record: DetectorInputRecord,
    output_path: str | Path,
    cuda_device: str | int | None = None,
) -> list[str]:
    if str(cuda_device) == "1":
        raise ValueError("GPU 1 is broken on this server; choose GPU 0, 2, 3, 4, 5, 6, or 7")
    values = _template_values(
        input_record=input_record,
        output_path=output_path,
    )
    try:
        rendered = command_template.format(**values)
    except KeyError as exc:
        raise ValueError(f"Unknown detector command placeholder: {exc}") from exc
    return shlex.split(rendered)


def run_detector_command(
    *,
    input_record: DetectorInputRecord,
    command_template: str | None,
    output_path: str | Path,
    method: str,
    cwd: str | Path | None = None,
    cuda_device: str | int | None = None,
    timeout_seconds: int | None = None,
) -> ProposalRecord:
    if input_record.failure_tag is not None:
        return ProposalRecord(
            scene_id=input_record.scene_id,
            scan_id=input_record.scan_id,
            target_id=input_record.target_id,
            method=method,
            input_condition=input_record.input_condition,
            proposals=[],
            failure_tag=input_record.failure_tag,
            metadata={
                "reason": input_record.metadata.get("reason", "detector input unavailable"),
                "detector_input_record": input_record.model_dump(),
            },
        )
    if input_record.pointcloud_path is None:
        return model_blocked_record(
            scene_id=input_record.scene_id,
            scan_id=input_record.scan_id,
            target_id=input_record.target_id,
            method=method,
            input_condition=input_record.input_condition,
            reason="detector input pointcloud_path is missing",
        )
    if not command_template:
        return model_blocked_record(
            scene_id=input_record.scene_id,
            scan_id=input_record.scan_id,
            target_id=input_record.target_id,
            method=method,
            input_condition=input_record.input_condition,
            reason="detector command template is not configured",
        )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = render_detector_command(
        command_template=command_template,
        input_record=input_record,
        output_path=out_path,
        cuda_device=cuda_device,
    )
    env = os.environ.copy()
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        timeout=timeout_seconds,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Detector command failed with exit code "
            f"{completed.returncode}: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if not out_path.exists():
        raise FileNotFoundError(f"Detector command did not write output JSON: {out_path}")

    record = load_detector_proposals_json(
        path=out_path,
        scene_id=input_record.scene_id,
        scan_id=input_record.scan_id,
        target_id=input_record.target_id,
        method=method,
        input_condition=input_record.input_condition,
    )
    record.metadata.update(
        {
            "detector_input_path": input_record.pointcloud_path,
            "detector_output_path": str(out_path),
            "detector_command": command,
            "cuda_device": str(cuda_device) if cuda_device is not None else None,
        }
    )
    return record


def _template_values(
    *,
    input_record: DetectorInputRecord,
    output_path: str | Path,
) -> dict[str, Any]:
    return {
        "scan_id": input_record.scan_id,
        "scene_id": input_record.scene_id,
        "target_id": "" if input_record.target_id is None else str(input_record.target_id),
        "input_condition": input_record.input_condition,
        "pointcloud_path": input_record.pointcloud_path or "",
        "output_path": str(output_path),
    }

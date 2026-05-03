"""Build the frozen 100-utt random fold for ScanRefer iteration.

Per CLAUDE.md memory: design iteration on the random100 fold first;
push to full 9508 only when smoke is good. Seed is intentionally fixed
so the fold stays identical across iterations and version comparisons
remain apples-to-apples.

Usage:
    python scripts/build_scanrefer_random100_fold.py

Output:
    tmp/scanrefer_artifacts/random100_sample_ids.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FULL_VAL = PROJECT_ROOT / "tmp/scanrefer_artifacts/full_val_sample_ids.json"
OUT = PROJECT_ROOT / "tmp/scanrefer_artifacts/random100_sample_ids.json"
SEED = 20260503  # NEVER change — fold must stay frozen across iterations
N = 100


def main() -> None:
    if not FULL_VAL.exists():
        raise FileNotFoundError(
            f"Missing {FULL_VAL}. Generate it via the full-val producer first."
        )
    full = json.loads(FULL_VAL.read_text(encoding="utf-8"))
    if len(full) < N:
        raise ValueError(f"full_val has only {len(full)} samples, need ≥ {N}")

    rng = random.Random(SEED)
    sample = rng.sample(full, N)
    sample.sort(key=lambda r: r["sample_id"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(sample, indent=2), encoding="utf-8")

    scenes = sorted({r["scene_id"] for r in sample})
    print(f"wrote {OUT} (n={len(sample)}, n_scenes={len(scenes)})")
    print(f"seed={SEED} (frozen — do not change)")


if __name__ == "__main__":
    main()

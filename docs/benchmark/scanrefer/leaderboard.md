# ScanRefer Public Leaderboard (reference)

Source: https://kaldir.vc.in.tum.de/scanrefer_benchmark/benchmark_localization
(test-server) and SeeGround Table 1 (val numbers from supervised SOTA).

## Test-server top-5 (official, Acc@0.5IoU Overall)

| Rank | Method | Unique@0.25 | Unique@0.50 | Multiple@0.25 | Multiple@0.50 | Overall@0.25 | Overall@0.50 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | UniVLG | 88.95 | 82.36 | 59.21 | 50.30 | 65.88 | 57.49 |
| 2 | Chat-Scene | 88.87 | 80.05 | 54.21 | 48.61 | 61.98 | 55.66 |
| 3 | ConcreteNet | 86.07 | 79.23 | 47.46 | 40.91 | 56.12 | 49.50 |
| 4 | cus3d | 83.84 | 70.73 | 49.08 | 40.00 | 56.88 | 46.89 |
| 5 | D-LISA | 81.95 | 69.00 | 49.75 | 39.67 | 56.97 | 46.25 |

## Validation top-5 (supervised, SeeGround Table 1)

| Rank | Method | Venue | Unique@0.25 | Unique@0.50 | Multiple@0.25 | Multiple@0.50 | Overall@0.25 | Overall@0.50 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | ConcreteNet | ECCV 2024 | 86.4 | 82.1 | 42.4 | 38.4 | 50.6 | 46.5 |
| 2 | 3D-VisTA | ICCV 2023 | 81.6 | 75.1 | 43.7 | 39.1 | 50.6 | 45.8 |
| 3 | MCLN | ECCV 2024 | 86.9 | 72.7 | 52.0 | 40.8 | 57.2 | 45.7 |
| 4 | G3-LQ | CVPR 2024 | 88.6 | 73.3 | 50.2 | 39.7 | 56.0 | 44.7 |
| 5 | EDA | CVPR 2023 | 85.8 | 68.6 | 49.1 | 37.6 | 54.6 | 42.3 |

## Zero-shot (Mask3D-pool) reference

| Method | Year | Unique@0.25 | Unique@0.50 | Multiple@0.25 | Multiple@0.50 | Overall@0.25 | Overall@0.50 |
|---|---|---:|---:|---:|---:|---:|---:|
| LLM-Grounder (998 sub-sample) | ICRA 2024 | - | - | - | - | 17.1 | 5.3 |
| ZSVG3D / GPT-4-Turbo | CVPR 2024 | 63.8 | 58.4 | 27.7 | 24.6 | 36.4 | 32.7 |
| CSVG (Mask3D) | BMVC 2025 | 68.8 | 61.2 | 38.4 | 27.3 | 49.6 | 39.8 |
| SeeGround / Qwen2-VL-72B | CVPR 2025 | 75.7 | 68.9 | 34.0 | 30.0 | 44.1 | 39.4 |
| VLM-Grounder (250 sub-sample) | CoRL 2024 | 66.0 | 29.8 | 48.3 | 33.5 | 51.6 | 32.8 |
| Z3D (Mask3D row) | 2026 arXiv | 82.3 | 74.8 | 51.5 | 45.7 | 58.9 | 52.7 |
| **Ours v2 (gpt-5.4)** | this work | **83.11** | **76.49** | **65.00** | **57.68** | **69.92** | **62.79** |
| Ours v1 (gpt-5.4) [†] | this work | 62.97 | 25.41 | 47.43 | 12.79 | 51.65 | 16.22 |

**v2 is the paper-comparable headline** — same v1 agent decisions
re-aggregated against ScanNet aggregation-derived GT (the bbox source
that Mask3D was trained against and that all other rows in this table
use). Camp-A SOTA across every column. See
[`v2_aggregation_gt_track_20260503.md`](v2_aggregation_gt_track_20260503.md).

[†] **v1 GT bbox not paper-comparable.** v1 uses Phase 8 GT-CG bbox
(~2× larger by volume than the ScanNet aggregation-based GT bbox the
other rows use). Oracle-picker ceiling on v1 is Acc@0.50 = 20.53 %;
v1 is preserved as audit trail. See
[`v1_oracle_analysis_20260503.md`](v1_oracle_analysis_20260503.md).

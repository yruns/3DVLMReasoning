# EmbodiedScan VG Leaderboard Notes

This file is a placeholder for public EmbodiedScan visual-grounding references.

The current tracked results are internal **GT-pool oracle smokes**:

| Method | Split | Proposal Source | n | Acc@0.25 | Acc@0.50 | mean IoU | Notes |
|--------|-------|-----------------|--:|---------:|---------:|---------:|-------|
| Ours v3 pack_v1 | ScanNet val smoke | GT annotations | 2000 | 89.15 | 89.15 | 89.20 | Projectable unique-target sweep; adaptive checkpointed run |
| Ours v2 pack_v1 | ScanNet val smoke | GT annotations | 67 | 94.03 | 94.03 | 94.08 | Corrected visible/projectable prep; 12 workers |
| Ours v1 pack_v1 | ScanNet val smoke | GT annotations | 68 | 91.18 | 91.18 | 91.18 | Pipeline verification only |

This is not a detector benchmark because every annotated instance is available
as a candidate proposal. Add public detector-based EmbodiedScan VG results here
only after checking the exact split, coordinate convention, and IoU threshold.

## Internal Detector-Pool Recall Studies

These are recall-only oracle upper bounds over detector proposals on the v3
frozen 2000-question fold. They are not Stage 2 visual-grounding scores.

| Method | Split | Proposal Source | n | Recall@0.25 | Recall@0.50 | mean max IoU | Notes |
|--------|-------|-----------------|--:|------------:|------------:|-------------:|-------|
| Ours v6 3-way | ScanNet val recall study | V-DETR + BIP3D + ConceptGraph union | 2000 | 77.35 | 53.20 | 49.29 | Keyframe-visible pool; ConceptGraph does not improve headline recall |
| Ours v6 ConceptGraph | ScanNet val recall study | ConceptGraph bbox_np proposals | 2000 | 0.35 | 0.00 | 0.88 | Keyframe-visible pool; axis-aligned `bbox_np` boxes |
| Ours v5 merged | ScanNet val recall study | V-DETR + BIP3D union | 2000 | 77.35 | 53.20 | 49.28 | Keyframe-visible pool; no NMS |
| Ours v5 BIP3D | ScanNet val recall study | BIP3D detector | 2000 | 72.50 | 43.30 | 42.19 | Keyframe-visible pool |
| Ours v5 V-DETR | ScanNet val recall study | V-DETR detector | 2000 | 40.80 | 33.05 | 29.50 | Keyframe-visible pool |

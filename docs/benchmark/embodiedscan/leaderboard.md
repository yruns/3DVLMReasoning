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

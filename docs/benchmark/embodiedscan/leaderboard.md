# EmbodiedScan VG Leaderboard Notes

This file is a placeholder for public EmbodiedScan visual-grounding references.

The current tracked result is an internal **GT-pool oracle smoke**:

| Method | Split | Proposal Source | n | Acc@0.25 | Acc@0.50 | mean IoU | Notes |
|--------|-------|-----------------|--:|---------:|---------:|---------:|-------|
| Ours v1 pack_v1 | ScanNet val smoke | GT annotations | 68 | 91.18 | 91.18 | 91.18 | Pipeline verification only |

This is not a detector benchmark because every annotated instance is available
as a candidate proposal. Add public detector-based EmbodiedScan VG results here
only after checking the exact split, coordinate convention, and IoU threshold.

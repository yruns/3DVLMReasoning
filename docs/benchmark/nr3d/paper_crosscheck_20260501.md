## Paper 1: ReferIt3D / ReferIt3DNet (ECCV 2020) — Q4, Q6 with citations

Working mapping used here: **Q4 = NR3D metric / leaderboard meaning**; **Q6 = object-candidate / proposal setup**.

**Q4.** ReferIt3D frames NR3D as GT-instance selection / classification, not IoU-threshold detection. The method section says it is cast into a **"classification problem"** that predicts the target object; Table 2 says the first row is **"accuracy on the Nr3D testing data"**. The reported NR3D-only ReferIt3DNet row is:

| Overall | Easy | Hard | View-dep. | View-indep. |
|---:|---:|---:|---:|---:|
| 35.6 +/- 0.7 | 43.6 +/- 0.8 | 27.9 +/- 0.7 | 32.5 +/- 0.7 | 37.1 +/- 0.8 |

Paper citation: Achlioptas et al., *ReferIt3D: Neural Listeners for Fine-Grained 3D Object Identification in Real-World Scenes*, ECCV 2020, Sec. 4 / Table 2: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460409.pdf

**Q6.** The candidate setup is GT segmented objects, not detector proposals. The paper assumes **"instance-level object segmentations"** and predicts among **"M segmented 3D instances"**. The official benchmark page describes the task as identification **"among all objects in a scene"**.

Benchmark citation: https://referit3d.github.io/benchmarks.html

## Paper 2: UniVLG (2025) — Q4, Q6 with citations

**Q4.** UniVLG separates GT leaderboard-style accuracy from Det IoU accuracy. Table 1 evaluates **"top-1 accuracy"** on the **"official validation set"** with GT and Det columns. Table 14 gives the ReferIt3D-style NR3D classification row for UniVLG:

| Overall | Easy | Hard | View-dep. | View-indep. |
|---:|---:|---:|---:|---:|
| 65.2 | 73.3 | 57.0 | 55.1 | 69.9 |

In the separate Det setup, UniVLG reports NR3D IoU-threshold results. For Sensor PC, UniVLG reports Acc@25 / Acc@50 / Acc@75 = **58.3 / 49.8 / 39.1**; UniVLG-3D-only reports **54.7 / 44.9 / 35.7**.

Paper citation: *Unifying 2D and 3D Vision-Language Understanding*, Table 1 / Table 14: https://openreview.net/pdf/6306d082de46d27c14c27436e4597009a5c8371a.pdf

**Q6.** UniVLG explicitly has two setups: Det, where methods **"do not have access to ground-truth 3D boxes"**, and GT, where methods use **"ground-truth 3D object proposals"**. For Det, a box is correct if its IoU with the GT box exceeds the threshold; for GT, correctness is selecting the GT object token.

## Q7 detection-mode NR3D survey

Yes, there are papers/supplements that report NR3D Acc@0.25 / Acc@0.50-style IoU results, but these are **not** the official NR3D public leaderboard columns. They appear in separate Det / 3DREC / 3DRES settings.

| Source | NR3D numbers | Protocol notes |
|---|---:|---|
| UniVLG, Table 1 | Sensor PC UniVLG: Acc@25 **58.3**, Acc@50 **49.8**. Sensor PC UniVLG-3D-only: **54.7 / 44.9**. Sensor PC 3D-VisTA re-eval: **42.1 / 37.4**. Sensor PC BUTD-DETR re-eval: **32.2 / 19.4**. Mesh PC PQ3D: **52.2 / 45.0**. Mesh PC 3D-VisTA: **47.7 / 42.2**. | Det = no GT proposals; paper says a predicted bbox is correct if IoU is above the threshold. Separate from GT Acc rows. |
| MCLN supplementary, Table 3 | 3DREC on NR3D: Acc@0.25 **59.82**, Acc@0.5 **51.38**. 3DRES on NR3D: Acc@0.25 **58.35**, Acc@0.5 **52.68**, mIoU **46.09**. | Supplemental says SR3D/NR3D **"does not require this dual process"** of detection and matching, so these are IoU localization/segmentation metrics rather than official candidate-selection leaderboard scores. Source: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06296-supp.pdf |
| BUTD-DETR, 3D-SPS, ViewRefer, VPP-Net, TSP3D | Not counted as direct both-threshold NR3D paper evidence. | BUTD-DETR reports NR3D Acc@0.25(det) only and ScanRefer both thresholds. 3D-SPS / ViewRefer / VPP-Net mainly report ReferIt3D-style target-selection accuracy or Acc@0.25 on NR3D. TSP3D paper uses Acc@0.25 for NR3D; its model card has Acc@0.5, but that is not a paper-direct table. |

## Cross-paper consistency check — do the two papers agree on Q4 and Q6? If they disagree, note the discrepancy.

They agree on the main protocol distinction.

For **Q4**, ReferIt3D reports NR3D as object-identification accuracy; UniVLG preserves that as the GT / detailed ReferIt3D row and keeps IoU Acc@25/50 in a separate Det table. Therefore the public NR3D leaderboard-style numbers should be read as classification / target-object selection accuracy, not Acc@0.25 / Acc@0.50 detection IoU.

For **Q6**, both papers support a GT object-candidate setup for leaderboard-style NR3D evaluation. ReferIt3D assumes segmented object instances; UniVLG calls this GT object proposals. The modern Det setting is a different protocol without GT boxes.

Potential wording discrepancy: ReferIt3D says "testing data", while UniVLG says "official validation set". I did not treat that as a metric disagreement; it is split-naming / paper wording unless the official code audit finds otherwise.

Important negative check: these paper citations do **not** support the claim that official NR3D leaderboard candidates are restricted only to same-class / target-type distractors. The cited evidence points to all GT object instances / proposals, with no category labels in the GT-box setup.

## Confidence — mark each finding as [CONFIRMED-PAPER-DIRECT-QUOTE] / [INFERRED-FROM-METHOD-DESCRIPTION] / [NOT-DOCUMENTED-IN-PAPER].

- ReferIt3D Q4 classification / accuracy: [CONFIRMED-PAPER-DIRECT-QUOTE].
- ReferIt3D Q6 GT segmented-instance candidate setup: [CONFIRMED-PAPER-DIRECT-QUOTE].
- "All objects in a scene" candidate wording: [CONFIRMED-PAPER-DIRECT-QUOTE] from official benchmark page, not paper PDF.
- UniVLG Q4 GT Acc vs Det Acc@25/50 separation: [CONFIRMED-PAPER-DIRECT-QUOTE].
- UniVLG Q6 Det vs GT proposal distinction: [CONFIRMED-PAPER-DIRECT-QUOTE].
- Q7 UniVLG Acc@25/50 NR3D Det numbers: [CONFIRMED-PAPER-DIRECT-QUOTE].
- Q7 MCLN Acc@0.25/0.5 NR3D numbers: [CONFIRMED-PAPER-DIRECT-QUOTE] for the numbers; [INFERRED-FROM-METHOD-DESCRIPTION] for treating them as detection-mode rather than official leaderboard classification.
- No support for "target-type-only public leaderboard pool" in these two papers: [NOT-DOCUMENTED-IN-PAPER].

## Addendum: UniVLG NR3D test prep

Repo skim found explicit filtering in UniVLG's public code (facebookresearch/univlg, commit `f86f0d5fa13864a518d7634e6fe3ad66f50d9f07`; repo: https://github.com/facebookresearch/univlg).

- Data source: `docs/DATA.md` downloads the ScanEnts3D NR3D CSV (`ScanEnts3D_Nr3D.csv`) and generates splits from it.
- Eval split naming: `docs/RUN.md` evaluates `nr3d_ref_scannet_anchor_val_single_batched`, backed by `ScanEnts3D_Nr3D_val.csv` in `load_sr3d.py`.
- Filter chain: `univlg/data_video/datasets/load_sr3d.py` keeps rows only when `line.get('mentions_target_class', True) is True`; for NR3D it additionally keeps rows only when `str(line['correct_guess']).lower() == 'true'`.

Interpretation: UniVLG's repo does apply both `mentions_target_class` and `correct_guess` filters at data-load time for NR3D-style CSVs. This is [CONFIRMED-REPO-DIRECT-CODE], not a paper-direct quote. I did not find equivalent explicit filter wording in the UniVLG paper PDF text; the paper documents the official validation split and GT/Det protocols but not these row-level filters.

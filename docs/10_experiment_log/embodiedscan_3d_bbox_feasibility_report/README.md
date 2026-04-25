# EmbodiedScan 3D BBox Feasibility Report

This folder is a self-contained report package for the EmbodiedScan RGB-D observation to 3D bbox feasibility study.

- Main report: [REPORT.md](REPORT.md)
- Visual dashboard: [dashboard.html](dashboard.html)
- Qualitative RGB/point-cloud gallery: [QUALITATIVE_GALLERY.md](QUALITATIVE_GALLERY.md)
- Qualitative gallery dashboard: [qualitative_gallery.html](qualitative_gallery.html)
- Figures: [resources/figures/](resources/figures/)
- Qualitative images: [resources/qualitative/](resources/qualitative/)
- Data snapshots: [resources/data/](resources/data/)

The report compares:

- 2D detection / segmentation driven ConceptGraph 3D proposals
- single RGB-D frame backprojection followed by V-DETR
- multi-frame RGB-D reconstruction followed by V-DETR
- camera-pose local raw ScanNet mesh crop followed by V-DETR
- full raw ScanNet scene mesh followed by V-DETR

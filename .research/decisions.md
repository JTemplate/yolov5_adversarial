# Research decisions

## 2026-08-25 — rebuild the four-vehicle study as Vehicle4 v2

- The legacy converter was verified to map `car+van`, `truck`, `bus`, and `pedestrian+people` while checkpoints/configs named the outputs `car`, `van`, `truck`, and `bus`.
- Legacy data, checkpoints, patches, and analyses are retained for audit but are not valid evidence for a four-vehicle claim.
- Vehicle4 v2 uses the immutable mapping `car(4)->0`, `van(5)->1`, `truck(6)->2`, and `bus(9)->3`, removes exact duplicate boxes, and keeps official validation isolated.
- The internal validation split is group-aware by VisDrone sequence prefix. All models receive the same ground-truth-anchored rendered images.
- YOLOv5s is the patch source; YOLOv5s, YOLOv5m, and fine-tuned Faster R-CNN are compared using relative mAP50-95 drop as the primary metric and ASR as secondary.

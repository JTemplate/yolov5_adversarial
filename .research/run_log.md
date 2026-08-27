# Run log

## 2026-08-25 — initial context compression

- Scanned README, dependency files, patch configs, experiment scripts, analysis CSVs, logs, run metadata, dataset directories, and transfer summaries.
- Wrote `project_manifest.yml`, `experiment_matrix.yml`, `data_dictionary.yml`, `decisions.md`, and `open_questions.md` under `.research/`.
- Preserved existing tracked modifications and untracked user files; no source or experiment code was changed.
- The `gpt_create/` directory was created for files produced by this request; `.research/` remains at the repository root because that is the required discovery path for future research skills.

## 2026-08-25 — Vehicle4 v2 experiment launch

- Confirmed the legacy class-mapping mismatch by comparing source annotations, generated labels, converter code, and checkpoint metadata.
- Generated `data/visdrone_vehicle4_v2/`: 5,821 train, 650 sequence-grouped internal validation, and 548 isolated official-validation images; removed one exact duplicate train box.
- Downloaded YOLOv5s/m COCO checkpoints and verified their SHA256 against the Ultralytics repository metadata.
- Launched YOLOv5s (GPU 0), YOLOv5m (GPU 1), and FasterRCNN-ResNet50-FPN (GPU 2) full fine-tuning.
- Started `experiments/vehicle4_v2/run_remaining_pipeline.py`, which records heartbeats and queues three-seed patch training, five-transform shared rendering, three-model evaluation, and aggregation.
- Environment and input provenance are recorded in `runs/vehicle4_v2/provenance.json`.
- Validated all `.research/*.yml` files with `yaml.safe_load` while the detector runs were active.
- Completed a one-image, 11-condition end-to-end smoke test covering shared rendering, YOLO inference, COCO scoring, ASR scoring, and final aggregation.
- Fixed repository-local imports in the rendering/evaluation entrypoints and made the concurrently shared COCO ground-truth JSON atomic.
- Added paired ASR confidence intervals with a three-level patch-seed, transform-seed, and image-id bootstrap; relative mAP-drop intervals remain at the patch/transform levels because COCO mAP is a dataset-level statistic.
- Live snapshot at `2026-08-25T18:40:37+08:00`: YOLOv5s completed epoch 28 with internal-validation mAP50-95 0.30933; YOLOv5m completed epoch 15 with 0.32662.
- FasterRCNN completed epoch 3 with internal-validation mAP50-95 0.21987, which was also its best checkpoint at this snapshot.

## 2026-08-25 — Vehicle4 v2 artifact relocation

- Established `experiment_data/vehicle4_v2/` as the canonical physical root for all current and subsequent Vehicle4 v2 experiment outputs.
- Briefly paused the three detector process groups and the pipeline monitor, moved the live run directory, created compatibility symlinks, and resumed all four process groups successfully.
- `experiment_data/vehicle4_v2/runs/` stores checkpoints, logs, patches, rendered inputs, predictions, provenance, and pipeline state; `experiment_data/vehicle4_v2/analysis/` stores final aggregate results.
- The audited source dataset remains at `data/visdrone_vehicle4_v2/` and was intentionally not moved.

# Open questions

- The repository does not contain a formal design brief or an explicitly stated research hypothesis; the `research_question` in the manifest is a working question inferred from the README and experiment structure.
- Confirm the exact provenance, download date, version, and license terms for the VisDrone archives and the four-class detector checkpoints before a publication or external release.
- The earlier `analysis/transformer_ablation/results_summary.csv` uses the typo `no_ratate`, while the current configuration is `ablation_no_rotate_10`; confirm that these artifacts refer to the same condition.
- Reconcile the duplicate/older run directories and determine which result is the canonical run for each experiment; the wrapper keeps timestamped outputs and the summary scripts select newest files.
- Decide whether detector-training variability also needs multiple detector seeds; the current controlled v2 comparison holds detector seed at 0 and varies patch/transform seeds.
- Decide whether the large dataset archives, checkpoints, logs, and generated run outputs should remain local, be moved to external artifact storage, or be added to a documented ignore/retention policy.
- Review the current working-tree modifications (`train_patch.py` and `adv_patch_gen/utils/patch.py`) and untracked experiment utilities before treating this snapshot as a clean baseline.

# Vehicle4 v2 experiment

This directory contains the reproducible rerun of the transfer experiment
after correcting the legacy VisDrone class mapping.

The immutable class mapping is:

- VisDrone 4 (`car`) -> v2 0
- VisDrone 5 (`van`) -> v2 1
- VisDrone 6 (`truck`) -> v2 2
- VisDrone 9 (`bus`) -> v2 3

The legacy labels and results are retained for audit only. They must not be
mixed with paths containing `vehicle4_v2`.

The pipeline writes durable state and logs under
`runs/vehicle4_v2/pipeline/`. A stage is skipped only after its expected
artifact exists and its completion marker has been written.

# GraphMoEAlignment

Research code for topology-guided mixture-of-anchor experts in multimodal glioma MRI semantic alignment.

Paper 4 adds an independent diagnostic metric-geodesic modality graph. It encodes MRI modalities separately, treats six modality-pair paths as graph edges, and fuses them under a data-induced metric instead of direct concatenation.

The Paper 2 implementation uses a dedicated `GliomaTopoMoENet` with shared semantic units, disease-anchor topology, family-balanced routing supervision, topology-refined anchor prototypes, checkpoint-specific evaluation, and deterministic intervention analysis. The legacy graph/diffusion model remains available for the Paper 1/3 configurations and the frozen TopoMoE v1 baseline.

## Repository scope

This repository contains source code, tests, server launchers, and documentation only. Patient data, metadata tables, checkpoints, experiment outputs, validation artifacts, and logs are intentionally excluded.

## Setup

```bash
pip install -r requirements.txt
```

Run the unit tests from the parent directory of this package:

```bash
python -m unittest discover -s glioma/tests -v
```

For the complete configuration, artifact protocol, ablations, and server commands, see [PROJECT_GUIDE.md](PROJECT_GUIDE.md).

## Full Paper 2 server run

```bash
DATA_ROOT=/path/to/UTSW-Glioma \
METADATA_TSV=/path/to/metadata.tsv \
bash run_server_paper2_topomoe_full.sh
```

The full launcher defaults to seeds `42 43 44`, evaluates the complete test split, and aggregates successful seed runs. Dataset paths and output locations remain local environment settings and are not committed.

## Paper 4 graph-evidence protocol

```bash
DATA_ROOT=/path/to/UTSW-Glioma \
METADATA_TSV=/path/to/metadata.tsv \
STAGE=screen \
bash run_server_paper4_graph_evidence.sh
```

The screening stage runs the non-collapsing Log-Euclidean graph, graph/manifold
ablations, and four matched published fusion baselines on seeds `42 43 44`.
After screening, run the same launcher with `STAGE=confirm`; it automatically
selects the best validation-set published baseline and adds seeds `45 46` for
the main model and key controls. See
[PAPER4_GRAPH_EVIDENCE.md](PAPER4_GRAPH_EVIDENCE.md) for background commands,
outputs, and evidence boundaries.

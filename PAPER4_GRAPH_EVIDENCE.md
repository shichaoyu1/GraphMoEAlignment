# Paper 4 Graph-Evidence Protocol

> This file preserves the legacy three-seed screen/confirm protocol. The ICLR
> protocol that separates split and model seeds is documented in
> [PAPER4_GUIDE.md](PAPER4_GUIDE.md) and launched by
> `run_server_paper4_iclr_evidence.sh`.

Paper 4 uses modality-specific MRI encoders and Log-Euclidean SPD descriptors.
The default `cross_budget` graph separates self mass from cross-node mass, so
distance softmax cannot collapse back to identity. Local modality graphs reserve
`0.35` cross-node mass. The upper region/family graph reserves `0.40`, including
`0.25` region-to-family mass when anchor families are enabled.

This is an architectural participation guarantee, not a guarantee of improved
retrieval. Performance claims are made only from the matched interventions and
multi-seed results.

Training topology is now selected with `--spd_local_topology` and
`--spd_upper_topology`. `--paper4_graph_intervention` is reserved for evaluation
of a trained checkpoint. Historical output directories remain readable.

## Screening

```bash
cd /root/autodl-tmp/GraphMoEAlignment

DRY_RUN=1 \
DATA_ROOT=/root/autodl-tmp/dataset/UTSW-Glioma \
METADATA_TSV=/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv \
GROUP_NAME=paper4_graph_evidence_v2 \
STAGE=screen \
bash run_server_paper4_graph_evidence.sh
```

Start the full screening stage in the background:

```bash
mkdir -p logs
nohup env \
  DATA_ROOT=/root/autodl-tmp/dataset/UTSW-Glioma \
  METADATA_TSV=/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv \
  GROUP_NAME=paper4_graph_evidence_v2 \
  STAGE=screen \
  bash run_server_paper4_graph_evidence.sh \
  > logs/paper4_graph_evidence_screen.log 2>&1 &
```

The screening stage runs ten variants on seeds `42 43 44`: the full SPD graph,
identity/uniform/local-only/no-anchor/Euclidean controls, latent concatenation,
HeMIS, GMU, and an MBT-style bottleneck baseline.

## Confirmation

After screening aggregation succeeds:

```bash
nohup env \
  DATA_ROOT=/root/autodl-tmp/dataset/UTSW-Glioma \
  METADATA_TSV=/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv \
  GROUP_NAME=paper4_graph_evidence_v2 \
  STAGE=confirm \
  bash run_server_paper4_graph_evidence.sh \
  > logs/paper4_graph_evidence_confirm.log 2>&1 &
```

Confirmation reuses seeds `42 43 44` and trains only seeds `45 46` for the six
core variants and the published baseline selected by mean validation mAP.

## Outputs

```text
output/server_runs/<GROUP_NAME>/<variant>/paper4/seed_*/
logs/server_runs/<GROUP_NAME>/<variant>/
output/server_runs/<GROUP_NAME>/aggregate/
```

The main SPD seed directories include `graph_role_metrics.json`, subgroup and
macro metrics in `test_metrics.json`, topology records, parameter counts, and:

```text
paper4_graph_participation.png
paper4_graph_interventions.png
paper4_manifold_ablation.png
```

Use `tail -f logs/paper4_graph_evidence_screen.log` or the corresponding confirm
log to monitor the outer launcher. Per-seed logs remain under the group log root.

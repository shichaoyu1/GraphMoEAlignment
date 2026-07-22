# Paper 4 Hierarchical SPD Manifold Fusion

The new Paper 4 main path uses a two-level heterogeneous graph instead of
feature-matrix concatenation:

```text
spatial MRI tokens
  -> modality-specific SPD descriptors
  -> Log-Euclidean local modality graphs (one per tumor region)
  -> region Frechet means
  -> region + pathology/molecular/residual upper graph
  -> tangent-space graph aggregation
  -> symmetric-vector readout for semantic retrieval
```

Pathology, molecular, and optional clinical family nodes are built only from
the train-set prototype bank. Patient grade and target labels are not model
inputs and are used only for training supervision or post-hoc figure labels.

## Full protocol

Run five matched variants with seeds 42, 43, and 44:

```bash
DATA_ROOT=/root/autodl-tmp/UTSW-Glioma \
METADATA_TSV=/root/autodl-tmp/UTSW_Glioma_Metadata-2-1.tsv \
bash run_server_paper4_manifold_full.sh
```

Use `DRY_RUN=1` to inspect all generated commands. Outputs are written below:

```text
output/server_runs/paper4_hierarchical_spd_v1/<variant>/paper4/seed_*/
output/server_runs/paper4_hierarchical_spd_v1/aggregate/
```

The aggregate manifest is marked `final_multiseed` only when all five variants
contain exactly three completed seeds. Existing vector-geodesic outputs remain
unchanged and serve as a legacy baseline.

# Idea2 投稿蓝图：2026-10-13 修订建议稿

当前优先级高于旧2026-08-05蓝图。旧版已原样备份于 `scratch/idea2_revision_20261013/PAPER2_NEUROCOMPUTING_BLUEPRINT.original.md`。本稿不代表新增实验已执行，也不保证特定期刊的创新性门槛已满足。

## Working title

**Factorized Anchor Retrieval under Incomplete Patient-Level Supervision**

## Central question

How much of the gain in family-factorized anchor retrieval comes from between-family allocation, within-family discrimination, and patient-dependent routing, and how sensitive are these effects to target construction and missing labels?

The empirical co-occurrence prior is a secondary comparison. Current evidence does not establish its added value over a capacity-matched unstructured mixture. Paper4 geometry remains outside this paper.

## Evidence available now

| Method / contrast | mAP or paired difference | Interpretation |
|---|---:|---|
| Direct | 0.6571 | Separately trained direct baseline |
| Unstructured family MoE | 0.7337 | Strong existing control |
| Prior-guided | 0.7343 | Complete training pipeline |
| Prior minus direct | +0.0773 [0.0716, 0.0830] | Pipeline gain; mechanism not yet isolated |
| Prior minus unstructured | +0.0006 [−0.0014, 0.0027] | Added prior benefit not established |

Five patient folds × three initialization seeds, 625 distinct held-out patients. Patient-aggregated paired bootstrap intervals. MRR does not clearly improve; grade 3 gain remains uncertain. Original identity node permutations are invalid controls; the replacement enumeration is post-hoc.

## Proposed manuscript structure

1. **Introduction:** incomplete, heterogeneous patient labels create a family allocation problem. Distinguish the new mechanism question from Paper1's task introduction.
2. **Related work:** local and multi-positive medical alignment, conditional retrieval, and routing. Compare MGCA/RadZero faithfully; patient label prototypes are not report embeddings.
3. **Task and protocol:** MRI units, label fields, prototype construction, region_rules, missing labels, 26-anchor gallery, negative candidates, worked synthetic patient example, splits and metric definitions. Use Hit@1 for the existing first-positive-hit metric.
4. **Factorized retrieval:** explicit family score equation; explain why gate-only changes preserve within-family ranks. State that this is a standard decomposition property, not a novel theorem.
5. **Experiments:** fixed full performance matrix, basic matched baselines, shared/separate projection × static/dynamic gate design, supervision sensitivity, grade and label-coverage strata, external validation only when available.
6. **Discussion:** distinguish score allocation, feature learning, supervision-conditioned behavior, and independent clinical correspondence. Include timing cost and limits of the small anchor gallery.

## Contribution wording permitted now

- A task-specific implementation of family-conditional label-prototype retrieval whose within-family ranking and between-family allocation can be inspected separately.
- Repeated patient-split evidence of improved mAP for the complete family mixture training pipeline, together with a matched control limiting attribution to the co-occurrence prior.
- An audit of target provenance and intervention validity that explicitly separates model dependence from clinical grounding.

Static-gate superiority/inferiority, supervision robustness, and external generalization are **pending questions**, not completed contributions. The present technical novelty remains limited until the controlled mechanism results distinguish the approach from simpler alternatives.

## Bounded execution order

1. Preserve the existing 45 runs; document targets and metrics; repair future nonidentity permutation audits without retraining the main matrix.
2. Add train-fold frequency and node-frequency controls, same-encoder global and label-classification baselines. Fit any score calibration using training/validation data only.
3. Compare shared/separate family projections × static node/patient-dynamic gates with topology absent and matched training. Frozen-direct calibration replay is a separate diagnostic.
4. On one fixed base architecture, compare region_rules/all_patient_anchors × family supervision on/off; disclose specialization penalties and remaining retrieval supervision.
5. Seek one independent cohort with compatible observed labels. Do not import Paper1 external numbers without their provenance. Keep internal robustness distinct from external transfer.

If resources require screening, use the prechosen seed42 across all five folds and pair it with seed42 references. Mark the work exploratory: existing held-out results have already informed the design. Expand only comparisons needed for the central question, and avoid selecting models from test results.

## Stop conditions

- If static calibration accounts for the gain and a dynamic benefit is not established, simplify the method and its title.
- If effects depend on the regional label rule, restrict conclusions to that weak-supervision protocol.
- If benefit remains concentrated in grade 4, report that limitation rather than choosing favorable cohorts.
- Do not add learned graphs, diffusion, residual semantics, or Paper4 geometry to rescue the original narrative.

## Sources and detailed revision draft

See [PAPER2_REVIEW_TRANSFER_20261013.md](D:/code/pythonProject/PygMonAI/glioma/PAPER2_REVIEW_TRANSFER_20261013.md) for the reviewer mapping, derivations, proposed controls, literature links, and reusable positioning paragraphs.

Verified results: [实验结果整理.md](D:/code/pythonProject/PygMonAI/glioma/output/sever_paper2+paper4_20260824alloutput/整理结果_20260908/实验结果整理.md).

## Implementation status: 2026-10-13

The supplemental protocol is implemented in the isolated `paper2_revision` package and new CLI entry points. Existing model, training, profile and server code is unchanged. It adds train-only frequency controls, static calibration, matched family-head factors, explicit supervision/candidate controls, published proxy-loss adaptations, protocol exports, nonidentity gate replay, paired aggregation and frozen external evaluation. These capabilities have been tested with synthetic NIfTI fixtures; new scientific results on the real cohort are not yet available.

See [实验执行说明](D:/code/pythonProject/PygMonAI/glioma/PAPER2_REVISION_EXPERIMENTS.md) and [代码审查记录](D:/code/pythonProject/PygMonAI/glioma/PAPER2_REVISION_CODE_REVIEW.md). MGCA/RadZero official-system reproduction is not claimed by the implemented loss adaptations.

# Idea2 Neurocomputing submission blueprint

## Working title

**Prior-Guided Mixture-of-Anchor Experts for Auditable Evidence Routing in Glioma MRI**

## Central claim

The paper presents a **prior-guided, auditable anchor-family evidence router**. It does not claim learned topology discovery, biological causality, or a clinically validated decision system.

The central learning-system question is:

> In a low-data medical retrieval setting, when does a train-fold disease co-occurrence prior improve family-level evidence allocation, and when does learned edge refinement add unnecessary flexibility?

The current evidence supports family-level functional dependence much more strongly than individual topology-edge dependence. The revision therefore treats fixed prior-guided routing as the main method and learned topology as a gated ablation.

## Current evidence and claim boundary

| Observation | Current estimate | Permitted interpretation |
|---|---:|---|
| Hybrid routed mAP | 0.6300 | Single-split/seed-aggregate starting point, not final main result |
| Prior-only routed mAP | 0.6298 | Learned edge correction has no demonstrated benefit |
| Identity/uniform/shuffled topology | approximately −0.004 mAP | Specific topology edges have weak test-time influence |
| Expert removal/wrong-family/family permutation | −0.106 to −0.281 mAP | Family routing is functionally used by the model |
| Residual usage | approximately 1.5% | Residual branch is unnecessary; removed in the revised profiles |
| Residual removal | ΔmAP = 0 | Confirms removal is safe for the observed run |
| Direct-best vs routed-best | 0.6813 vs 0.6300 mAP | The old routed head has a material optimization deficit |
| Routed-best epoch | epoch 1 for all three seeds | Joint training/checkpoint selection is mismatched to routed retrieval |

Edema–molecular and Enhancing/Core–pathology patterns are reported only as **stable model-routing patterns** until they survive target-policy, supervision, and label-completeness audits. Interventions are described as **model-internal sanity checks** or an **intervention audit**.

## Contributions, ordered for Neurocomputing

1. An anchor-family hierarchical mixture architecture that decomposes MRI-to-anchor retrieval into auditable evidence-family routes.
2. A train-fold-only disease co-occurrence prior used as fixed routing context, with a capacity-matched unstructured MoE control.
3. A patient-clustered intervention and supervision-sensitivity audit that separates architectural behavior from target construction and missing-label effects.
4. An empirical design result for low-data learning: fixed family structure may be sufficient, while learned anchor-edge correction is retained only if it clears a pre-registered gain threshold.

The fourth contribution is a conditional result, not a universal theorem.

## Journal fit

[Neurocomputing](https://www.sciencedirect.com/journal/neurocomputing) explicitly covers neural architectures, learning methods, network dynamics, pattern recognition, and image processing. The manuscript should therefore lead with the learning mechanism, controlled architecture comparison, and audit protocol. Glioma MRI is the test bed rather than the sole novelty.

Two same-journal comparisons must be explicit:

- Yu et al., [Multimodal multitask similarity learning for vision language model on radiological images and reports](https://www.sciencedirect.com/science/article/pii/S0925231225006903), *Neurocomputing* 636 (2025) 130018, DOI: 10.1016/j.neucom.2025.130018. M2SL learns relational and knowledge-driven semantic similarity for radiology image–report retrieval. The present work differs by routing region-level MRI evidence through an explicit anchor-family hierarchy and auditing the routing mechanism.
- Wang et al., [HMP-Net: A hierarchical multi-prior network for brain tumor segmentation integrating physics, topology, and tumor dynamics](https://www.sciencedirect.com/science/article/pii/S0925231226012245), *Neurocomputing* 691 (2026) 133827, DOI: 10.1016/j.neucom.2026.133827. HMP-Net uses Betti/morphological topology and physical/dynamical priors for segmentation. Here, “topology” is a train-fold anchor co-occurrence transition prior; it is not spatial homology, Betti topology, or tumor morphology.

## Manuscript structure

### 1. Introduction

- Problem: region-level MRI evidence is aligned to heterogeneous pathology and molecular anchors, but a single similarity head does not expose which evidence family supports a retrieval.
- Gap: medical retrieval work emphasizes representation alignment; prior-guided mixture models rarely test whether the prior itself matters, whether learned refinement is needed, or whether apparent specialization is induced by supervision.
- Proposal: a fixed-prior family router with intervention and supervision audits.
- Contributions: use the four contributions above, keeping performance claims conditional on the non-inferiority analysis.

### 2. Related work

1. Medical image–text and image–concept retrieval, including M2SL.
2. Mixture-of-experts routing and structured/gated retrieval.
3. Knowledge-guided learning in medical imaging.
4. Topology in medical imaging, separating graph/co-occurrence topology from Betti/morphological topology as used by HMP-Net.
5. Auditing interpretability claims under supervision leakage and missing labels.

### 3. Method

#### 3.1 Task and anchor vocabulary

- Define each patient as the independent unit.
- Define Core, Edema, and Enhancing queries.
- Construct the anchor vocabulary from the training fold only.
- State target-policy alternatives (`region_rules`, `all_patient_anchors`).

#### 3.2 Direct anchor aligner

- Shared MRI encoder and prototype bank.
- Direct cosine-similarity retrieval head.
- This is the capacity reference for non-inferiority.

#### 3.3 Family router

- Partition anchors into pathology, molecular, and clinical families when available.
- Remove the residual family.
- Compute family gates and within-family conditional anchor probabilities.
- Describe `direct logits + log family gate` as the conservative fallback score, not as the default unless the independent expert projection misses the non-inferiority bound.

#### 3.4 Train-fold prior

- Build patient-level anchor co-occurrence counts on the training fold only.
- Row-normalize to obtain the fixed prior.
- Define empirical, uniform, and deterministic random training priors.
- Define `prior_plus_learned` and its regularization, but mark it as an ablation unless it passes the learned-topology gate.

#### 3.5 Three-stage optimization

1. Train the direct encoder and prototype bank to direct validation optimum.
2. Freeze encoder/prototypes; train router and experts with direct-to-routed score distillation.
3. Jointly fine-tune at 0.1× learning rate with routed validation mAP early stopping.

#### 3.6 Audit protocol

- Training controls: empirical, uniform, and random priors.
- Model interventions: topology identity/uniform/shuffle; uniform, patient-shuffled, and node-shuffled routing; expert removal; anchor-family permutation.
- Supervision audits: target policy and family-balanced route supervision on/off.
- Missingness audits: IDH, MGMT, 1p/19q availability and molecular-complete subset.
- Routing contrasts: within-patient differences between habitats.

### 4. Experiments

#### 4.1 Cohort and preprocessing

- Report inclusion/exclusion, patient count, MRI sequence completeness, habitat availability, grade distribution, and all molecular-label missingness.
- State ethics approval and de-identification details using verified institutional wording.

#### 4.2 Cross-validation

- Five-fold patient-level split with fixed fold seed.
- Test fold `k`, validation fold `k+1`, remaining folds for training.
- Three initialization seeds for `direct_only`, `unstructured_family_moe`, and `prior_guided_router`.
- Vocabulary, prior, normalization statistics, and supervision statistics are derived independently within each training fold.

#### 4.3 Baselines

- `direct_only`: same encoder/prototype bank; no router.
- `unstructured_family_moe`: same expert capacity; raw family context, no topology propagation.
- `prior_guided_router`: fixed empirical train-fold prior.
- `prior_plus_learned`: learned correction ablation.
- Uniform and random prior training controls.

#### 4.4 Metrics and statistics

- mAP, MRR, and Recall@1 are computed per query and averaged within patient first.
- Concatenate held-out predictions across folds and average repeated initialization results within patient.
- Use 10,000 paired patient bootstrap resamples for method differences and 95% confidence intervals.
- Report parameter count, approximate Conv/Linear FLOPs, peak GPU memory, and per-patient latency.
- Primary non-inferiority margin: −0.01 mAP for prior-guided routed versus direct-only.

### 5. Results

1. Main patient-level comparison and efficiency.
2. Direct-to-routed non-inferiority and optimization trajectory.
3. Fixed-prior versus unstructured/controlled priors.
4. Learned correction versus fixed prior.
5. Intervention audit.
6. Within-patient routing contrasts and missingness/supervision sensitivity.

Do not open Results with qualitative routing heatmaps. Lead with the main patient-level estimates and confidence intervals.

### 6. Discussion

- Interpret family-level dependence and weak edge-level sensitivity separately.
- If fixed prior and unstructured MoE are indistinguishable, narrow the claim to auditable decomposition rather than structured-routing benefit.
- Explain why the old jointly trained routed head selected epoch 1 and how staging/distillation changes the optimization problem.
- Discuss supervision-conditioned specialization before any domain interpretation.
- Limitations: single center, finite label availability, rule-based target construction, anchor vocabulary dependence, no external cohort, retrieval surrogate rather than clinical outcome.

### 7. Conclusion

Conclude with the learning-system finding supported by the final gates. Do not conclude that the model discovers glioma biology.

## Main figures and tables

- **Figure 1:** MRI habitats → anchor families → train-fold prior-guided router → anchor retrieval. Visually distinguish the fixed prior from learned parameters.
- **Figure 2:** Direct, unstructured MoE, prior-guided router, and optional learned correction. Show patient-level 95% CIs and a companion efficiency panel.
- **Figure 3:** Within-patient routing contrasts with 95% CIs, stratified by label availability and sensitivity run.
- **Figure 4:** Training controls and model interventions. Use effect-size/CI plots rather than isolated bars.
- **Table 1:** Cohort, labels, availability, and missingness by fold.
- **Table 2:** Five-fold × seed main comparison, patient-level estimates and paired differences.
- **Table 3:** Prior, topology, target-policy, supervision, and fallback scoring ablations.
- **Supplement:** learned new-edge stability, per-fold results, full intervention subgroups, run configuration, and compute environment.

## Submission gates

| Claim | Required evidence | Action if failed |
|---|---|---|
| Routed model is the main performance method | Lower 95% CI of routed − direct mAP > −0.01 | Present routing as an audit/calibration layer, not a performance method |
| Structured routing is effective | Prior-guided beats unstructured and uniform/shuffled controls with paired CI excluding 0 | Use “auditable family decomposition”; remove structure-benefit claim |
| Learned topology is a main contribution | Mean gain ≥ 0.005, lower CI > 0, same direction in every fold | Move learned correction to supplement |
| Habitat-specific evidence allocation | Same direction and CI excluding 0 under alternate target policy, supervision off, and complete-label subset | Use “supervision-conditioned auditable decomposition” |

Until all necessary gates are available, the internal editorial status remains **Reject/Resubmit—premature**. A complete statistical and supervision audit can raise it to a submission-ready manuscript likely to require major revision.

## Required declarations

Add verified, author-approved text for:

- Ethics approval and consent/waiver.
- Data availability and access restrictions.
- Code availability with a frozen release identifier.
- CRediT authorship contributions.
- Funding.
- Declaration of competing interest.
- Generative AI disclosure consistent with Elsevier policy at submission time.

Do not invent approval numbers, funders, repositories, or author roles.

## Implemented experiment artifacts

- `run_server_paper2_neurocomputing.sh`: five-fold × three-seed main matrix and optional full sensitivity matrix.
- `cli/train_semantic_alignment.py`: controlled profiles, staged optimization, distillation, train-time prior controls, and patient CV.
- `cli/audit_paper2_neurocomputing.py`: 10,000-resample patient bootstrap, routing contrasts, figures, CSV, and submission gates.
- `semantic/paper2_statistics.py`: patient-first metrics and paired inference.
- `training/efficiency.py`: parameter, FLOP approximation, memory, and latency profiling.

The original research note remains useful as the development history. This blueprint is the submission-facing narrative and should be used as the source of truth for the revised manuscript.

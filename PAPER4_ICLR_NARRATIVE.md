# Paper 4 ICLR Narrative Blueprint

## Working Title

**Topology Is Not the Message: Disentangling Geometry, Communication, and Co-Adaptation in Multimodal Fusion**

备选标题：

1. **Do Learned Modality Graphs Learn Structure? A Diagnostic Study of Geometry and Communication in Multimodal Fusion**
2. **When Graph Sensitivity Misleads: Geometry, Communication Budgets, and Identifiability in Multimodal Fusion**

默认使用第一个标题。它先提出一般机器学习问题，再暗示反直觉结果，不把医学应用或新模块名放在标题中。

## One-Sentence Thesis

> Downstream gains and test-time graph sensitivity do not establish topology recovery, because representation geometry, communication support, and edge allocation can independently affect performance and can become co-adapted during training.

中文解释：论文不是证明“图没用”，而是证明当前常见证据不足以回答“学到的图是否正确”，并提供一套可证伪的诊断协议。

## Research Questions

- **RQ1:** When a learned modality graph improves downstream performance, which gain comes from representation geometry, enabling communication, or allocating edge weights?
- **RQ2:** Does performance degradation under a test-time graph intervention imply that the learned topology is correct or identifiable?
- **RQ3:** Under what data-generating conditions can downstream learning recover a planted modality graph?
- **RQ4:** Can the diagnosis produce a simpler fusion rule without sacrificing performance?

## Abstract Skeleton

```text
Learned modality graphs are often evaluated by downstream performance and
interpreted through their edge weights. These criteria conflate three effects:
the geometry used to represent each modality, the availability and budget of
cross-modal communication, and the allocation of that budget across edges.

We introduce a diagnostic framework that evaluates these effects with matched
retraining controls, test-time interventions, fixed-split stability, and planted
graph recovery. We establish two limitations of common evidence: uniform-budget
message passing can be exactly invariant under mean pooling, and sensitivity to
a graph replacement can coexist with superior performance after retraining on
the replacement graph.

Across synthetic data, AV-MNIST, CMU-MOSEI, and a multimodal MRI retrieval task,
we find [PUBLIC-DATA RESULT]. In the MRI task, SPD moment geometry provides the
largest reproducible gain, while a uniformly budgeted graph outperforms learned
edge allocation by [CONFIRMED VALUE]. These findings motivate SPD-UB, a simpler
fusion rule that removes locally redundant graph learning and uses fixed-budget
upper-level communication.

Our results argue for evaluating learned modality graphs as latent structures,
not only as downstream network components, and provide concrete tests for when
their edges can support structural interpretation.
```

方括号内容只有在 factorial confirmation 完成后才能填。摘要不使用 “first”, “novel”, “SOTA” 或 “real topology” 等无法由当前证据支持的词。

## Contributions

1. **Diagnostic factorization.** We separate multimodal graph fusion into representation geometry, communication support/budget, and edge allocation, with matched retraining and intervention metrics.
2. **Formal limitations.** We show a pooling invariance result for uniform-budget graphs and give a constructive co-adaptation counterexample demonstrating that intervention sensitivity is insufficient for topology identification.
3. **Cross-domain evidence.** We test planted graph recovery and downstream utility on synthetic data, AV-MNIST, CMU-MOSEI, and multimodal MRI retrieval under fixed split/model seed protocols.
4. **Simplified consequence.** We derive SPD-UB, which uses SPD moment representations, local identity pooling, and optional uniform-budget upper communication instead of learned local edges.

贡献顺序固定为“问题与诊断 → 理论限制 → 广泛证据 → 简化方法”。SPD-UB 是诊断结果，不应被写成先拍脑袋提出、再用实验包装的新模块。

## Conceptual Decomposition

令 `S(g, c, a)` 表示由 geometry `g`、communication policy `c` 和 allocation `a` 训练得到的模型分数：

- **Geometry gain:** `S(SPD, c, a) - S(Euclidean, c, a)`
- **Communication gain:** `S(g, communicate, fixed) - S(g, identity, none)`
- **Allocation gain:** `S(g, communicate, learned) - S(g, communicate, uniform)`
- **Intervention cost:** `S(A,A) - S(A,B)`
- **Retrained advantage:** `S(B,B) - S(A,A)`
- **Co-adaptation gap:** `S(B,B) - S(A,B)`

这里最重要的设计是 matched retraining。只报告 `S(A,A) - S(A,B)` 会把参数与图的共同适应误写成 topology evidence。

## Formal Results

### Proposition 1: Uniform-Budget Pooling Invariance

For `M` available modalities and cross-modal mass `α`, define

```text
Aα = (1 - α)I + α/(M - 1)(11ᵀ - I).
```

`Aα` is symmetric and doubly stochastic. For linear message passing followed by mean pooling,

```text
(1/M)1ᵀAαH = (1/M)1ᵀH.
```

**Use in the paper:** this proposition does not claim that all uniform GNNs are useless. It applies to the current single linear local propagation and pooling operator. Nonlinearities, edge-dependent transformations, repeated layers, or non-doubly-stochastic masks may break the equality.

### Proposition 2: Intervention Sensitivity Does Not Imply Identifiability

Construct a two-modality linear predictor with topology-conditioned parameters. There exist data distributions and topologies `A` and `B` such that

```text
S(A,A) > S(A,B),
S(B,B) > S(A,A).
```

The first inequality shows test-time sensitivity. The second shows that the replacement topology is better after retraining. Hence sensitivity of parameters trained with `A` cannot identify `A` without assumptions linking the downstream optimum to the data-generating graph.

**Proof strategy:** give an explicit two-feature least-squares or logistic example in the appendix, then reproduce it numerically in the synthetic benchmark. Do not call this a general impossibility theorem.

## Method: SPD-UB

**SPD-UB = SPD Moment Representation with Uniform-Budget Communication.**

1. Each modality encoder returns a token sequence.
2. Valid tokens form a trace-normalized covariance descriptor.
3. The descriptor is mapped to the Log-Euclidean tangent space.
4. Local modalities use identity/mean pooling; the redundant uniform local propagation is removed.
5. When groups or semantic supports exist, an upper graph allocates a fixed cross-node budget uniformly by allowed edge type.
6. Symmetric vectorization and a small readout produce the task representation.

医学版本的 support nodes 来自 train-set pathology/molecular prototype families。公开数据不强行制造语义锚点；`G=1` 时退化为 SPD moment pooling，保持方法定义诚实。

## Nine-Page Paper Architecture

| Section | Working budget | Purpose and evidence |
|---|---:|---|
| Abstract | 180–220 words | 问题、分解、两条限制、跨数据发现、SPD-UB |
| 1. Introduction | 1.0 page | 从 learned graph interpretation 风险切入；给出 co-adaptation reversal；列贡献 |
| 2. When Does a Modality Graph Mean Structure? | 1.0 page | latent graph、multimodal fusion、graph diagnostics；明确研究缺口 |
| 3. A Diagnostic Factorization | 1.5 pages | 三因素定义、六个指标、seed protocol、Propositions 1–2 |
| 4. SPD-UB | 1.0 page | SPD moments、local invariance、upper fixed-budget communication、复杂度 |
| 5. Experimental Design | 1.0 page | synthetic、AV-MNIST、MOSEI、UTSW；matched controls 与统计协议 |
| 6. Results | 2.3 pages | 真值恢复、公开数据、co-adaptation、效率、医学 stress test |
| 7. Discussion and Limitations | 0.8 page | 可解释边的条件、negative result 边界、单中心数据限制 |
| 8. Conclusion | 0.3 page | 回答 RQ，不重复所有结果 |

Appendix 收录：完整命题证明、数据生成过程、全部 seed 结果、缺失模态曲线、超参数、伦理与数据可用性声明。

## Claim–Evidence–Qualification Map

| Claim | Required evidence | Current evidence | Qualification |
|---|---|---|---|
| Geodesic fidelity is not task utility | matched Euclidean/geodesic comparison | UTSW vector run supports | 不推广到所有测地模型 |
| SPD geometry is a primary gain source | same graph and budget, SPD vs Euclidean | 两轮 UTSW 支持 | 公开数据待完成 |
| Learned allocation is not established | learned vs uniform retraining | UTSW prototype: uniform +0.0256 | 完整 baselines/seed confirmation 未完成 |
| Intervention sensitivity is insufficient | formal counterexample + empirical reversal | UTSW reversal 已出现 | 不否认干预作为诊断的一部分 |
| Learned edges are identifiable in some regimes | planted edge AUROC/AUPRC across regimes | 未完成 | 必须先通过 synthetic falsification gate |
| SPD-UB is a competitive simplification | public + UTSW results, parameters/FLOPs | UTSW 原型与参数量支持 | 不提前声称普遍优于 learned graph |

## Experimental Matrix

### Synthetic

- Modalities: `M ∈ {4,6}`
- Graphs: chain, star, two-community
- Regimes: geometry-only, exchangeable, topology-relevant
- Missingness: `0%, 25%, 50%`
- Randomness: 5 data seeds × 5 model seeds
- Primary structural metrics: edge AUROC, edge AUPRC, edge Spearman
- Primary task metric: accuracy

**Falsification rule:** if learned allocation cannot recover planted edges in the topology-relevant regime, diagnose the parameterization before making claims about real-data topology.

### AV-MNIST

- Official split; five model seeds.
- Primary metric: accuracy.
- Stress tests: image/audio token noise and one-modality dropout.
- Main comparison: SPD learned, SPD uniform/identity, Euclidean learned, concat/GMU/MBT-style where feasible.

### CMU-MOSEI

- Official aligned split and token masks; five model seeds.
- Primary metric: binary F1; secondary metrics: accuracy, MAE, Pearson correlation.
- Stress tests: text/audio/video missingness and Gaussian feature noise.

### UTSW MRI Retrieval

- Screening: split seeds 42/43/44 × model seed 101.
- Confirmation: the same splits × model seeds 101/102/103.
- Primary metric: mAP; secondary: R@1, MRR, pathology/molecular macro mAP.
- Core models: learned-both, identity-both, uniform-both, identity-local/learned-upper, SPD-UB, matched Euclidean, best validation-selected published baseline.

## Main Figures and Tables

1. **Figure 1 — The diagnostic trap.** Geometry, communication, and allocation flow into downstream performance; test-time intervention and retraining form two distinct paths.
2. **Figure 2 — Synthetic identifiability phase diagram.** Task performance versus edge recovery across regimes, graphs, and missingness.
3. **Figure 3 — Cross-dataset decomposition.** Geometry, communication, and allocation gains with paired 95% confidence intervals.
4. **Figure 4 — Co-adaptation and efficiency.** `S(A,A)`, `S(A,B)`, `S(B,B)`, parameter count, FLOPs, and wall time.
5. **Table 1 — Public benchmark main results.** AV-MNIST and MOSEI.
6. **Table 2 — UTSW stress test.** Overall and target-family metrics; medical results do not appear in Table 1.

不要把 MRI 病例图放在 Figure 1。若保留医学可视化，只放 appendix 或最后一个结果子节。

## Reviewer Objection Matrix

| Likely objection | Pre-emptive response | Rebuttal evidence to prepare |
|---|---|---|
| “This is only an application paper.” | 前三组实验为 synthetic、AV-MNIST、MOSEI；UTSW 是最后的 stress test | 公共数据主表与通用接口 |
| “Uniform beats learned because the learner is weak.” | topology-relevant synthetic 必须显示何时可恢复；同时比较容量与优化 | edge recovery、training curves、bias initialization |
| “Intervention sensitivity is still useful.” | 同意其诊断价值，但它不能单独证明 identifiability | Proposition 2 与 matched retraining |
| “Uniform and identity are algebraically identical.” | local 层确实等价；贡献来自 upper communication，因此提出 SPD-UB | Proposition 1、独立 local/upper controls |
| “Results are split-dependent.” | split seed 与 model seed 完全解耦，报告 hierarchical bootstrap | 3×3 UTSW factorial table |
| “No SOTA.” | 论文目标是诊断结构解释与收益来源，不是堆叠模块 | 官方评审口径、参数/FLOPs 与 falsifiable findings |
| “The medical graph has no ground truth.” | 不声称它是真实图；只报告稳定性与下游作用 | synthetic truth recovery + qualified UTSW analysis |

## Related-Work Spine

按问题组织，而不是按应用罗列：

1. Learned latent graph structure and the distinction between predictive fit and structural recovery.
2. Wrong-structure optima and identifiability limits.
3. Multimodal fusion benchmarks and robustness to missing/noisy modalities.
4. SPD/covariance representations and Log-Euclidean computation.
5. Post-hoc graph explanations and intervention-based diagnostics.

已核验的起点：

- [Learning Latent Graph Structures and their Uncertainty, ICML 2025](https://openreview.net/forum?id=TMRh3ScSCb)
- [Learning Large DAGs is Harder than you Think, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2858e880333b3cd64f8192f13ddcca2f-Abstract-Conference.html)
- [NetInfoF, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/220165f9c7f51163b73c8c7fff578b4e-Abstract-Conference.html)
- [MultiBench, NeurIPS Datasets and Benchmarks 2021](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/37693cfc748049e45d87b8c7d8b9aacd-Abstract-round1.html)

正式 Related Work 写作前仍需进行一次 citation audit，补齐 DOI/BibTeX，且不能用这四篇替代系统检索。

## Venue Transfer Matrix

| Component | ICLR | ICML | NeurIPS |
|---|---|---|---|
| Opening | learned representation 是否被误读为结构 | latent structure 的 estimand 与 matched estimator | learned-graph evaluation practice 的广泛风险 |
| Main emphasis | 反直觉现象、诊断框架、representation insight | 命题、统计协议、条件化结论 | benchmark、negative result、robustness、use-inspired impact |
| Method framing | SPD-UB 是诊断后得到的简单模型 | SPD-UB 是 factorization 的受约束估计器 | SPD-UB 是强而简单的 benchmark baseline |
| Results order | synthetic → public → UTSW | theorem/synthetic → estimation → real data | benchmark breadth → robustness → real data |

三会共享方法、实验、图表和约 90% 正文；只重写 title、abstract、Introduction、Related Work 顺序和 contribution wording。任何转投都先核对当年匿名、页数、预印本和 supplementary 政策。

## Writing Style Rules

- Introduction 第一段讨论 learned modality graphs 的证据问题，不介绍 glioma。
- 每个强主张附近同时出现 matched evidence 和限制条件。
- 固定使用 `topology recovery`、`communication budget`、`edge allocation`、`co-adaptation`，不要循环替换同义词。
- 不写 “clearly demonstrates”, “proves the learned graph”, “universally”, “first-ever”。
- 结果部分先报 paired differences 和 CI，再讨论均值。
- Discussion 主动承认：single-center medical data、feature-level public scaffold、有限 graph family、命题适用的线性层条件。
- 最终稿必须包含 Limitations、Data Availability、Ethics、Conflict of Interest、Funding、CRediT、AI-use disclosure。

## Backward Schedule

ICLR 2027 日期尚未公开。内部先沿用上一周期的 9 月 19 日 abstract、9 月 24 日 paper 节点：

- **Aug 5–10:** 完成接口、synthetic scaffold、证据总账和 narrative blueprint。
- **Aug 11–20:** 跑 synthetic 与 AV-MNIST，完成 topology falsification gate。
- **Aug 21–Sep 2:** 跑 MOSEI 与 UTSW factorial confirmation。
- **Sep 3–10:** 统计、主图、错误分析和 claim freeze。
- **Sep 11–18:** 完整英文初稿、内部双盲审阅、citation audit。
- **Sep 19–23:** 摘要锁定、复现实验核查、匿名与伦理材料检查。

若 9 月 2 日前公开数据没有形成一致证据，立即放弃“uniform is better”的强叙事，保留更稳健的中心结论：**graph sensitivity is not topology identification**。

最后更新：2026-08-05。

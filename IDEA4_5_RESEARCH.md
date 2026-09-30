# Idea4.5：模态可用性改变时的学习路径干扰

日期：2026-09-24。性质：文献支持的研究设计；撰写时尚未实现，也没有新实验结果。后续第一阶段代码见 [Idea4.5 指南](IDEA4_5_GUIDE.md)，冒烟测试不能视为研究结果。本文由 AI 辅助检索、原文核查与方法审查形成。

**续审修订：**已补读 CMoB 与 ICYM²I 的正式全文方法。新增检索发现，TAG（NeurIPS 2021）已定义虚拟更新后的跨任务损失变化，Gradient Routing（2024 作者预印本）已实现前向不变的反向通路门控。因此本文中的更新审计与梯度门控均降为已有工具，不能单独作为创新。当前候选问题进一步收紧为“可用模态上下文是否改变同一共享路径的学习作用”。具体四组干预、组合基线及停止判据见 [Idea4.5 v2 补充核查与最小实验协议](IDEA4_5_PROTOCOL.md)。

## 1. 建议收敛成的科学问题

**在共享参数的多模态学习中，不同可用模态组合之间何时产生正迁移、机会成本或实际训练侵蚀？能否通过配对训练干预定位这些效应，并利用图上的通信路径减少有害更新，同时保留有益的跨模态学习？**

项目代号：**Idea4.5**。工作标题：**When Modality Availability Changes the Effect of Shared Learning Paths**。此为研究问题标题，不预告已经获得正向结果。

这里暂按“同一个模型同时学习模态齐全与部分缺失的样本/视图”理解用户的问题。若实际指不同数据集分别训练，首先要排除域差异；若所有模态都存在而只是信息量强弱不同，则更接近已有 balanced multimodal learning。三个设置不可混写。

研究单位从“强模态与弱模态”推进到“可用模态组合之间的训练影响”。但这一单位变化本身不能声称首次提出：ANGA 已研究完整与重建样本之间的优化失衡，BALM 已研究不均衡缺失率。配对虚拟更新和反向门控也都有先例。候选增量在于检验**新增模态上下文与共享学习路径的交互**，以及是否能在保持前向信息可用的条件下，以更少的路径修改改善稀缺风险与丰富效用的权衡。其原创性与有效性均待验证。

## 2. 与原 Idea4 的真实连接

已检查本项目的 PAPER4_AUTHORITY_GUIDE.md、PAPER4_MANIFOLD_GUIDE.md、PAPER4_GRAPH_EVIDENCE.md、PAPER4_ICLR_NARRATIVE.md、authority 主稿以及两轮结果报告。

- 既有实现含模态特异 SPD 描述、Log-Euclidean 坐标、局部与上层图通信，可以作为新研究的结构化模型实例。
- 现有 authority 方案的固定参数前向通路保护，不包含跨训练步骤的共享参数非干扰保证。新问题正好可以研究这一尚未覆盖的训练层面。
- UTSW 结果支持当前协议中 SPD/Euclidean 对照的收益，但均匀图优于学习边分配；不能预设“更复杂的边学习更好”。[本地 UTSW 报告](D:/code/pythonProject/PygMonAI/glioma/reports/paper4_confirmation_20260908/RESULTS.md)
- authority v2 合成结果中，Euclidean 在所报告的两个任务、两种联合投影架构上均优于 SPD；ACF 相比同信息同投影基线也没有稳定、实质性的普遍预测优势。[本地 v2 解读](D:/code/pythonProject/PygMonAI/glioma/reports/paper4_authority_v2_20260914/RESULTS_INTERPRETATION.md)

因此可以保留 graph 与 manifold，但应该让它们接受独立消融。新方向不需要给一般模态人为指定临床式 authority；若任务没有外生资格规则，保护对象应来自可用组合的训练风险，而不是由“谁模态多/谁预测强”决定资格。

## 3. 近三年已核实文献与创新边界

检索时间范围为 2023-09-24 至 2026-09-24，核心直接相关文献落在 2024—2026。以下主表均有正式会刊归属依据；作者版全文用于补充阅读时，不以 arXiv 上传年份替代正式刊年。全文阅读聚焦问题设置和机制，没有重跑这些论文的代码。

阅读深度：MMPareto、Data Remixing、BALM、ANGA、MD²N、ARL 已读取官方主文方法；Representation Collapse 与 TPAMI modulation 使用已确认归属的作者全文补读。续审已补读 CMoB 正式主文 §3.2–3.4、Eq.(1)–(5)、Algorithm 1，以及 ICYM²I 正式全文 §3–4 和限制部分；此前两篇全文受限的缺口已补上。CMoB 评估当前预测的模态贡献并选择增强数据，不能误述为有限步跨组合风险审计；ICYM²I 要求 MAR 与 positivity，不提供任意 MNAR 校正保证。

| 文献 | 原文支持的结论 | 对本计划的约束 |
|---|---|---|
| Wei & Hu, **MMPareto: Boosting Multimodal Learning with Innocent Unimodal Assistance**, ICML 2024 | 多模态目标与单模态辅助目标可能梯度冲突，处理方向与幅度差异。[官方论文集](https://proceedings.mlr.press/v235/wei24d.html) | “多/单模态目标难度不同、梯度冲突”已经有研究；通用梯度投影不能单列为创新。 |
| Wei et al., **On-the-Fly Modulation for Balanced Multimodal Learning**, TPAMI 47(1), 469–485, 2025；2024 已接收/上线 | OPM/OGM 从前向与反向调节优势模态对联合优化的支配。[DOI](https://doi.org/10.1109/TPAMI.2024.3468315)；[作者全文](https://arxiv.org/html/2410.11582v1) | 它研究模态学习不均衡；不能把“强者压制弱者”作为新发现。 |
| Ma, Chen & Deng, **Improving Multimodal Learning Balance and Sufficiency through Data Remixing**, ICML 2025 | 讨论优化速度与路径差异，通过数据重组缓解学习不充分及跨模态干扰。[官方论文集](https://proceedings.mlr.press/v267/ma25c.html) | 需要比较简单的数据组织策略，避免多算若干次梯度被包装成机制创新。 |
| Chaudhuri et al., **A Closer Look at Multimodal Representation Collapse**, ICML 2025 | 在论文设定下，从特征纠缠与秩瓶颈解释模态坍缩，并提出 basis reallocation。[官方论文集](https://proceedings.mlr.press/v267/chaudhuri25a.html)；[作者全文](https://arxiv.org/html/2505.22483v2) | 共享容量/表征纠缠是替代解释；不能仅看到负梯度内积就宣称找到了唯一机制。 |
| Wang et al., **CMoB: Modality Valuation via Causal Effect for Balanced Multimodal Learning**, NeurIPS 2025 | 用样本级模态贡献估计指导动态增强与平衡。[官方论文集](https://papers.nips.cc/paper_files/paper/2025/hash/76a0f4215050c7d4e88c03cb8a11fee6-Abstract-Conference.html) | “反事实/因果贡献估计 + 弱模态增强”已有直接近邻。本文应考察更新后的跨组合风险，而非仅更名预测贡献。 |
| Nguyen et al., **BALM: A Model-Agnostic Framework for Balanced Multimodal Learning under Imbalanced Missing Rates**, CVPR 2026 | FCM 校准特征；GRM 调节模态梯度的幅度与方向，直接面向不均衡缺失率。[官方论文集](https://openaccess.thecvf.com/content/CVPR2026/html/Nguyen_BALM_A_Model-Agnostic_Framework_for_Balanced_Multimodal_Learning_under_Imbalanced_CVPR_2026_paper.html) | 必须纳入最接近方法；其模态梯度调节与本计划跨可用组合的有限步后果有差别，但差别的价值需要证明。 |
| Guan, Huang & Yang, **Anchor-Guided Gradient Alignment for Incomplete Multimodal Learning**, CVPR 2026 | ANGA 研究严重缺失时重建样本主导优化、削弱完整样本；课程选锚后按梯度方向调节。[官方论文集](https://openaccess.thecvf.com/content/CVPR2026/html/Guan_Anchor-Guided_Gradient_Alignment_for_Incomplete_Multimodal_Learning_CVPR_2026_paper.html) | 已涉及不同完整性组之间的优化失衡；而且伤害方向与“丰富侵蚀稀缺”相反，故方向必须测量。 |
| Choi et al., **ICYM²I: The illusion of multimodal informativeness under missingness**, ICLR 2026 | 缺失机制及环境变化会偏置模态信息增益估计；在其假设下采用逆概率加权校正。[官方论文集](https://proceedings.iclr.cc/paper_files/paper/2026/hash/883cf7782b794759b98dd604c4adb5a1-Abstract-Conference.html) | 从完整样本模拟缺失不能自动代表真实缺失群体；需要区分优化效应和选择偏差。 |
| Dai et al., **A Study of Dropout-Induced Modality Bias on Robustness to Missing Video Frames for Audio-Visual Speech Recognition**, CVPR 2024 | 在 AVSR 设置中研究 dropout 改善缺失鲁棒性却损害完整输入表现的权衡。[官方论文集](https://openaccess.thecvf.com/content/CVPR2024/html/Dai_A_Study_of_Dropout-Induced_Modality_Bias_on_Robustness_to_Missing_CVPR_2024_paper.html) | 必须同时报告丰富与稀缺条件性能，不能靠忽略丰富输入获得表面鲁棒性。 |

另核查 **Unbiased Missing-modality Multimodal Learning**（Dai et al., ICCV 2025）：MD²N 主要处理缺失模态生成偏差，主文设定训练模态完整、推理缺失，不能用作天然缺失训练中跨组侵蚀的证据。[官方全文](https://openaccess.thecvf.com/content/ICCV2025/papers/Dai_Unbiased_Missing-modality_Multimodal_Learning_ICCV_2025_paper.pdf)

另一项必要的反向证据是 Shicai Wei、Chunbo Luo、Yang Luo 的 **Improving Multimodal Learning via Imbalanced Learning**（ICCV 2025）。其 ARL 在特定融合及偏差—方差分析中允许不均衡贡献，提醒我们将目标设为减少有害迁移，而非强制所有模态等权、等速或等性能；其受限分析不能外推为任意网络下不均衡都更优。[官方全文](https://openaccess.thecvf.com/content/ICCV2025/papers/Wei_Improving_Multimodal_Learning_via_Imbalanced_Learning_ICCV_2025_paper.pdf)

**当前可说：**相关问题已经被多个角度研究，值得在训练操作的可干预性和图路径定位上寻找更窄的增量。**当前不可说：**首次发现模态失衡、首次用反事实平衡多模态、首次处理完整/缺失样本梯度冲突，或者无人研究这一问题。

## 4. 把“稀缺收敛慢”和“被侵蚀”拆成可检验命题

令可用模态集合为 s，丰富集合为 r，配对实验采用 s⊂r。这里“丰富”只表示可用集合包含关系，不意味着其数据质量或实际信息量必然更高。模型为 fθ(X_s,s)。在统一潜在人群 P 上定义人工可用性干预风险：

\[
R_s(\theta)=\mathbb E_{(X,Y)\sim P}\,\ell(f_\theta(X_s,s),Y).
\]

真实缺失人群的条件风险 E[ℓ|S=s] 是另一个量；若 S 与环境、标签或质量有关，不能不加条件地将两者等同。

三个待检验命题：

1. **收敛速度差异：**稀缺组到达预先指定且可达的性能水平，需要更多计算或有效更新。
2. **负迁移终点差异：**共享混合训练比合理的组专属参考训练，在同一稀缺评估条件上表现更差。
3. **实际训练侵蚀：**来自另一组的一次/一段更新，使固定稀缺评估风险上升。

它们不能互相替代。不同组合的原始 loss、准确率差距还包含信息差异。例如在相同分布、嵌套观测与最优对数损失预测器下：

\[
R_s^*-R_r^*=H(Y\mid X_s)-H(Y\mid X_r)
=I(Y;X_{r\setminus s}\mid X_s)\ge0.
\]

这是条件熵恒等式的直接推导，不是新的理论贡献，也不推出“丰富组优化更快”。真实实验中未知 Bayes 风险应保持未知；组专属长训模型只能提供 reference risk，不能冒充 R_s*。

## 5. 把反事实放在训练操作上

**来源与新颖性修正：**以下跨组合更新损失变化借鉴已有 task affinity 思路。TAG 的有向分数 Z(r→s)=1−R_s(θ′_r)/R_s(θ)，在同样更新与评估设置下，与本文 Δabs 满足 Δabs=−R_s(θ)Z；并不是新指标。TAG 还比较了 training/validation loss 审计。[TAG，NeurIPS 2021，§4.1、§5.2](https://proceedings.neurips.cc/paper/2021/file/e77910ebb93b511588557806310f78f1-Paper.pdf)。该文超出近三年范围，作为不能遗漏的基础先例单列，不冒充近三年论文。

### 5.1 最小配对审计

在训练时刻 t 保存完整状态 Z_t：参数、优化器状态、调度器、归一化缓冲及需要复用的随机性。取同一训练样本批次 B 的丰富/稀缺视图，从同一个 Z_t 分叉：

\[
Z'_r=U(Z_t;B_r),\qquad Z'_s=U(Z_t;B_s).
\]

另用不参与该步梯度的训练内审计子批次 B_a，在相同稀缺视图上计算：

\[
\Delta^{abs}_{r\to s}
=\widehat R_s^{a}(\theta'_r)-\widehat R_s^{a}(\theta_t),
\]

\[
\Delta^{rel}_{r:s\to s}
=\widehat R_s^{a}(\theta'_r)-\widehat R_s^{a}(\theta'_s).
\]

- Δabs>0：该更新后审计风险实际变差。
- Δrel>0：丰富视图更新比稀缺视图更新更不利；可能两者都改善，因此不能单独叫侵蚀。
- 同时报告反方向 s→r，以及不同训练阶段和缺失组合；不先假定单向伤害。

这一实验估计的是**给定训练状态和样本群体时，训练操作改变的局部算法效应**。它不是识别真实世界模态生成因果图，也不自动证明长期性能差异来自同一机制。

操作要求：初步机制实验关闭 dropout 或使用按模块独立的随机流；同一个种子未必使不同分支消耗的随机数一一对应。审计时使用一致评估模式。先冻结归一化缓冲以隔离参数更新，再单独检查缓冲变化。AdamW 的实际位移包括动量、预条件与衰减；若要把数据效应和优化器背景效应分离，需增设保持相同优化器背景的中性对照。

若审计结果用于选择更新，B_a 属于训练资源。真正验证与测试数据不得参与门控或更新选择。用独立测试样本评估已冻结的方法。

### 5.2 图的实质用途：定位训练干扰路径

保留现有模态节点图。先不同时新增一个“组合图”网络：所有非空可用集合最多有 2^M−1 个，直接逐组合建模可能失控。少模态任务可完整枚举；模态多时先预定少量嵌套组合及关键缺失情况。

对消息路径 e 进行第二层配对干预：

1. 同一丰富视图、相同参数下正常前向和反向，得到更新 U_on。
2. 保持消息数值与前向预测相同，仅在该路径对消息使用 stop-gradient，得到 U_block。
3. 比较两次更新后的同一稀缺审计风险。

\[
I_{e,r\to s}
=\widehat R_s^{a}(\theta'_{on})-\widehat R_s^{a}(\theta'_{block}).
\]

I>0 表示，在这次配对训练干预下，允许该反向路径比阻断它更不利。共享参数仍可经其他路径收到梯度，故这不是“该模态所有影响的自然因果效应”。要区分两类干预：删除前向边改变了信息及 softmax 归一化；阻断反向路径改变学习过程。两者不能混成同一边效应。

这是 graph 比简单拼接更值得研究的地方：是否提供可定位、可干预的学习通路。仅有 attention 可视化不足以建立这一点。必须加入随机阻断、同数量路径阻断、decoder-only 干预及无图模型对照。

前向不变、反向按数据相关 mask 路由本身也已有直接先例：[Gradient Routing，§3，2024 作者预印本](https://arxiv.org/html/2410.04332v1#S3)。本次不将其标为已接收顶会。本文若只组合 TAG 与梯度路由，不能以“反事实”更名支撑方法创新；v2 用观测上下文×通路门控的四组交互对照进一步检验科学假设。

### 5.3 保留流形，但不把它当自动防侵蚀机制

可沿用每模态 token 的均值增强 SPD 描述：

\[
S_m=\begin{bmatrix}C_m+\mu_m\mu_m^\top&\mu_m\\\mu_m^\top&1\end{bmatrix}
+\epsilon I,\qquad z_m=\operatorname{symvec}(\log S_m).
\]

在固定维度的 log 坐标中做现有图通信；缺失模态用 availability mask 表示。若只有一个 pooled token，普通样本协方差退化，不能假装得到了充分的二阶统计；优先使用 patch/frame/word tokens，或明确仅均值特征与正则化项起作用。

SPD 在这里的假设是协方差/矩结构可能提供适合某类观测的表示，不是因果识别条件。Log-Euclidean SPD 对应的 log 坐标可视为欧氏几何，不能靠“流形有曲率”解释实际并未采用的机制。神经网络参数的梯度约束，也不会因前向特征是 SPD 就自动成为 Riemannian optimization。

最小独立消融：SPD/Euclidean × graph/匹配融合结构 × 干扰控制/无控制。保留相同输入信息、表示维度、编码器与合理参数预算；必要时再区分均匀通信与学习边权。

## 6. 可以推导什么，哪些不能写成新定理

对 SGD 的局部分析，令训练梯度为 g_r，审计风险梯度为 g_s^a，则：

\[
\Delta^{abs}_{r\to s}\approx-\eta\langle g_s^a,g_r\rangle.
\]

在统一人口风险的理想化两组模型中，若混合方向为 g=π_r g_r+π_s g_s：

\[
R_s(\theta-\eta g)-R_s(\theta)
\approx-\eta[\pi_s\|g_s\|^2+\pi_r\langle g_s,g_r\rangle].
\]

只有负的跨组项足以压过稀缺组自己的下降项时，才预测混合一步实际伤害。π 是采样/损失权重，不能偷换为模态数量。丰富组收敛较快也不意味着其梯度始终更大。

以上来自一阶展开，是机制分析起点，不是创新定理。若目标 L_s-smooth：

\[
R_s(\theta-\eta d)-R_s(\theta)
\le-\eta g_s^\top d+\tfrac{L_s\eta^2}{2}\|d\|^2.
\]

因此 g_sᵀd≥0 只是一阶保护，不能保证有限步长无损，更不能保证非凸长期收敛。Adam 应检查实际参数位移，而不是未经预处理的梯度内积。两个组梯度完全相反时，不存在同时严格下降方向；任何保护机制都应允许松弛、记录停滞及丰富组代价。

较有意义的理论目标应是：在明确受限模型中，刻画何时组合间干扰出现；给出带估计误差与有限步长项的审计可信条件；或证明某种路径控制比全局梯度控制有更小代价。尚未证明，不作为已有贡献。

## 7. 方法候选：先证明审计有价值，再增加控制器

**第一版以诊断为主。**用现有 SPD 图，在预先指定的少量检查点执行状态分叉与路径审计，构建“来源组合 × 受影响组合”的更新效果矩阵。先验证有限步测量是否比梯度 cosine、模态置信度、缺失率更能预测实际损害。

**第二版才学习控制。**设 a 是允许的图反向路径门控，不改变该次前向消息值，Θ⁺(a) 为候选一次真实优化更新后的参数。一个可执行的试验目标为：

\[
\min_{a\in\mathcal A}\sum_s w_s\widehat R_s^a(\Theta^+(a))
+\lambda\sum_{s\in\mathcal S_{protect}}q_s
[\widehat R_s^a(\Theta^+(a))-\widehat R_s^a(\theta)-\varepsilon_s]_+
+\rho C(a).
\]

其中 w_s 是预先声明的目标组权重，q_s 控制保护强度，ε_s 是容许风险变化，C(a) 可惩罚过度阻断或额外计算。候选集合必须包含正常更新，并考虑只用稀缺训练子步的参考。第一版可用小规模离散候选，不必立即实现昂贵的二阶元学习。

这只是候选目标，不是已验证算法或原创性主张。审计批次反复选动作会过拟合，需轮换训练内审计数据；对多步收益仍要独立测量。若通用梯度投影或增加稀缺重放就达到同样效果，应接受更简单方案。

不要将丰富与稀缺表示/预测强制完全相等：丰富观测可能提供真实的独有信息和协同信息。需要保护的是稀缺条件的学习质量，而不是抹掉新增信息。

## 8. 最小有判别力的实验路线

### 阶段 A：先建立现象，暂不新建大模型

1. 合成二/三模态任务：控制信噪比、共享/独有/协同标签信息、可用组合频率和噪声。采用相同潜在样本的嵌套掩码，加入不应产生干扰的正迁移对照及容易产生竞争的设置。
2. 一个非临床三模态任务：可用 CMU-MOSEI 的文本/音频/视频，检查全模态及预定义子集；沿用标准数据划分。
3. 固定编码器先检查融合层机制；随后解冻编码器，判断结果是否仍成立。冻结实验的训练速度只代表融合头，不代表整个模型训练效率。
4. 对比混合训练、稀缺专属训练、组均衡采样、稀缺额外重放。不同训练模型的终点差异只是参考，配对一步干预才直接检验局部更新效应。

### 阶段 B：验证替代解释与控制器

核心基线：现有 SPD 均匀图/学习图、modality dropout、按可用组合均衡采样、组风险重加权、通用梯度冲突处理、MMPareto、BALM、CMoB；ANGA 在采用相同 VLP/重建条件时比较，不能随意删除其重建环节后仍称完整复现。

续审新增最强组合基线：TAG 式组间信号重加权，以及 TAG 式伤害评分＋现有反向梯度门控。比较须匹配审计计算预算；只胜过无控制模型或 PCGrad/BALM，并不能证明超越这些现成构件的自然组合。

按架构兼容性区分原方法复现与改编版本。对旧通用优化基线可在后续实现前补充其原始文献；本次近三年主表不替代完整 related work。

需要排除：少看了稀缺样本、仅计算量增加、表征容量瓶颈、batch/归一化效应、权重衰减/动量、缺失引起消息预算变化、蒸馏目标不适用，以及丰富/稀缺组原本来自不同人群。

### 阶段 C：决定是否扩展

增加图文任务，例如 MM-IMDb；医疗 MRI 作为额外外部场景。将随机可用性实验与质量/环境相关缺失分开报告。对 MNAR 设置，只在已知生成机制的合成实验中做可核验识别结论；真实数据未经额外假设不声称已消除选择偏差。

先采用统一任务标签；如果缺失动作会改变标签定义，本方案的同标签配对就不适用。测试时不把真实不可见模态用于门控、补全真值或子集选择。

### 必须报告的量

| 维度 | 指标与解释 |
|---|---|
| 学习过程 | 每可用组合的曲线；到达预注册可达阈值的步数/时间；未达到记为未达到；必要时报告学习曲线面积 |
| 更新后果 | Δabs、Δrel 的分布与置信区间；正/负迁移比例；反向 s→r；有限步实测与一阶预测的一致性 |
| 终点效用 | 组合均值、最差组合、各具体组合结果；丰富组的损失不能省略 |
| 图机制 | 前向删除/反向阻断分开；定向阻断 vs 随机等预算阻断；共享 head/encoder 的替代路径 |
| 几何机制 | SPD vs 相同矩描述的 Euclidean 对照；统计量构造与矩阵 log 的作用分开解释 |
| 成本 | 同步数、同样本接触量、同计算预算三个视角；额外审计耗时和显存 |
| 不确定性 | 多训练种子及独立数据种子/适当划分；同一检查点的多次审计不是独立模型复现 |

## 9. 可以继续投入的证据门槛

- 若控制样本曝光和信息差异后，没有稳定的正 Δabs，应改写为学习效率/机会成本问题，不强称侵蚀。
- 若负效应主要来自 shared head，而图路径干预没有额外解释力，图可继续保留为模型实现，但不能列为核心机制贡献。
- 若一阶梯度量已充分解释有限步审计，且带来的方法效果相同，不能仅凭“反事实”名称主张创新。
- 若 SPD 无稳定增益，保留它作为受控实现或适用场景结果；不声称流形必要。
- 若保护稀缺组必然降低丰富组表现，应报告 Pareto 权衡，避免只选某个有利组。

最值得先做的图：**同一训练状态下，不同可用组合的更新对其他组合造成的风险变化矩阵**，按训练早/中/晚期分别绘制，随后对最稳定的有害格子做图路径干预。它比先增加反事实生成网络更直接检验目前的科学命题。

## 10. 检索与证据限制

检索入口：CVF Open Access（CVPR/ICCV）、PMLR（ICML）、NeurIPS 与 ICLR 正式论文集、OpenReview、IEEE DOI 元数据与作者公开全文。题名、作者、会刊年逐篇核对；未把搜索摘要中的 under review 当已接收。部分站点直接访问有 403、验证页或 PDF 大小限制，使用官方索引、官方 PDF 下载或已确认归属的作者全文补查。

主要检索词：multimodal missing modalities optimization imbalance；imbalanced missing rates；complete incomplete gradient conflict；multimodal representation collapse；counterfactual causal modality valuation；missingness information gain；graph manifold missing modality。另按 BALM、ANGA、CMoB、MMPareto、Data Remixing、ICYM²I 等精确题名追踪。

这是聚焦研究设计的定向文献核查，不是穷尽数据库的 PRISMA 系统综述；不报告未经记录的命中总数，不宣称已排除所有相似工作。没有发现与上述完整候选链条相同的方法，不等于证明其原创。本计划中的公式定义、路径干预目标和实验判据是建议方案，未在本次对话中通过新训练验证。

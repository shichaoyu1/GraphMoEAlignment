# Paper2 修订实验：独立入口与执行协议

版本：2026-10-13。代码补充完成不等于真实数据实验已完成。当前新训练管线的验证使用合成NIfTI夹具，数值不得用于论文。

## 模块边界

新增 `paper2_revision/` 负责目标审计、损失、家族评分、训练和评估；新CLI负责单次训练、矩阵计划、配对汇总和冻结外部测试。旧 `train_semantic_alignment.py`、配置、TopoMoE、Paper1–4模型与服务器脚本均无代码修改。复用的既有模块只作为数据读取、训练折词表和ROI编码器依赖，不修改其全局行为。

新结果使用独立目录和 `paper2_revision_v1` 输出格式，不生成旧汇总器会自动搜集的 `patient_level_records.json`。单次训练拒绝写入已存在目录。矩阵执行前检查所有输出冲突；`--skip_completed` 只跳过参数、源码和记录的环境版本一致且已完成的运行，未完成运行不会被悄悄覆盖。

## 已实现内容

| 模块 / 实验 | 行为 |
|---|---|
| frequency / node_frequency | 仅用训练患者拟合标签频率与节点条件频率，不读测试标签拟合模型 |
| direct | 原ROI区域encoder架构、可学习标签原型、集合正例NLL；独立训练参照 |
| global | 同样数量和结构的encoder均看完整ROI，平均表征用于三个query；容量受控但不是单encoder全球方法 |
| multilabel | 每个已观测且规则选中的标签字段作类别交叉熵；未知字段无损失，字段内log-softmax映射到共同gallery |
| mean_positive_proxy | mean-positive log-probability的标签原型适配；区分逐正例优化与集合总质量优化 |
| proxy_anchor_multilabel | CVPR2020 Proxy Anchor损失的多标签适配，采用稳定logsumexp；同encoder和标签原型，非官方数据/训练方案复现 |
| calibration | 读取旧direct_only检查点，冻结encoder与原型，只训练静态global/node gate；共享原始logits |
| factorized | 共享原始评分/独立family线性投影 × 静态node/患者dynamic gate；无topology |
| supervision | region_rules/all_patient_anchors × 显式family gate监督on/off；基础结构固定为separate+dynamic |
| gate replay | 穷举5个非恒等节点排列；患者置换保证无固定点；保存每种实际映射。均为固定专家回放 |
| 评估 | full-gallery为共同主要终点；observed-fields为另一候选协议的诊断，不混成同一任务。输出患者聚合mAP/MRR/Hit@1、grade分层与macro、family内AP和有效分母 |
| 配对统计 | 要求相同fold/seed、全部患者划分、query正例与候选库；先患者内聚合节点/seed，再患者bootstrap。拒绝静默取交集 |
| 外部测试 | 从已完成的新训练读取模型与词表，冻结推理，无外部拟合/校准/选epoch；拒绝源队列患者ID重叠，记录词表外标签与有效覆盖 |

“SOTA比较”仍须依据同任务结果判断。新增两项已发表损失的适配基线，不能称2026年最先进系统，也不能代替MGCA/RadZero完整影像报告模型的直接复现。当前数据提供标签原型，缺少相同输入条件的报告文本；文稿须公开这个适配边界。

参考：[Proxy Anchor，CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.html)，[官方损失实现](https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020)。[SupCon，NeurIPS2020](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)用于说明mean-positive形式的来源；本适配使用query-to-proxy而非原始batch样本对比，不叫SupCon原系统复现。

## 必须公开的协议

正例构造复用旧函数，保证同一policy行为一致：Core→Grade/1p19q/Type，Edema→IDH/Type/Grade，Enhancing→Grade/MGMT/Type；若所选字段全缺失，旧规则回退为患者全部可用标签，新审计显式计数。anchor来源是标签字段—取值，prototype为可学习参数。

默认 `full_gallery` 将所有原型作为竞争候选；未知标签与规则排除的已知标签不会自动从分母删除。`observed_fields` 显式将候选限制到规则选中的已观测字段，训练和诊断各自声明使用的协议。它使用标签可用性定义评价候选集，是监督评估协议，不是MRI-only部署推理。

`family_supervision=off` 只关闭等家族gate交叉熵；检索监督和默认 `lambda_within=0.3` 仍存在。若检验纯集合检索驱动的路由，再显式传 `--lambda_within 0`。新管线没有specialization、balance、distillation或图正则，权重0在审计中记录。它与旧完整MoE训练配方不同，所以不能把新增factorized和旧0.7343直接当作只改一个因素的对照。

阶段为direct→router→joint；router阶段冻结encoder/原型及BatchNorm统计，joint恢复训练。每阶段保存validation最佳模型，最终从router/joint的validation最佳中选择，测试只在选择后评估。没有有效验证集会报错，不用测试集替代。

provided segmentation定义区域，空mask保留并计数；不能因此声称区域自主发现。`--segmentation_source`填写真实来源，默认 `not_documented` 会保存在协议文件中而不自动猜测。缺失/OOV/空目标/回退/区域规则排除标签及覆盖数均记录。协议依然有小标签gallery、患者级弱监督与缺失机制未识别的限制。

## 服务器执行

从 `glioma` 目录运行，脚本自动使父目录可导入该包：

```bash
# 默认只打印10次低成本频率基线的计划，不启动训练。
bash run_server_paper2_revision.sh

# 先实际跑便宜对照；使用独立输出目录。
EXECUTE=1 GROUP=cheap bash run_server_paper2_revision.sh

# 原45次主矩阵中的direct_only根目录，仅作为只读参照。
EXECUTE=1 GROUP=calibration REFERENCE_ROOT=/path/to/direct_only bash run_server_paper2_revision.sh

# 同协议的4格机制对照，共5折×1seed×4=20次。
EXECUTE=1 GROUP=mechanism bash run_server_paper2_revision.sh

# 补另外3格监督消融，region_rules+on复用mechanism/separate_dynamic。
EXECUTE=1 GROUP=supervision bash run_server_paper2_revision.sh

# 按需运行全局/分类/直接参照以及已发表损失基线。
EXECUTE=1 GROUP=baselines bash run_server_paper2_revision.sh
EXECUTE=1 GROUP=published bash run_server_paper2_revision.sh
```

可以传 `DATA_ROOT`、`METADATA_TSV`、`RUN_ROOT`、`SEEDS`、阶段epoch、batch和分割来源。先seed42，扩展seed43/44时只执行必要的组。`GROUP=screen`为14配置×5折=70次单seed探索计划，不默认启动，也不将其包装为先验确定的确认实验。

复用旧direct检查点时，新入口验证train/val/test患者、完整词表、输入维度、ROI、影像配准偏好、温度、目标规则，以及检查点内词表。`REFERENCE_ROOT`使各新组沿用旧splits；改变为all_patient_anchors时仅复用划分，重新训练表征，避免不匹配目标的warm start。reference文件从不写入。

单次观测字段消融示例，从包父目录运行：

```bash
python -m glioma.cli.train_paper2_revision --data_root /path/to/UTSW-Glioma \
  --out_dir /path/to/new_run --method factorized --projection separate --gate dynamic \
  --target_policy region_rules --family_supervision off --lambda_within 0 \
  --candidate_policy observed_fields
```

配对比较示例（从包父目录运行）：

```bash
python -m glioma.cli.aggregate_paper2_revision \
  --baseline /path/to/shared_node --candidate /path/to/shared_dynamic \
  --out_file /path/to/new_audit/shared_dynamic_vs_node.json

# 冻结校准的直接评分回放和校准评分在同组患者上配对。
python -m glioma.cli.aggregate_paper2_revision \
  --baseline /path/to/calibration_node --baseline_score direct \
  --candidate /path/to/calibration_node \
  --out_file /path/to/new_audit/calibration_vs_frozen_direct.json
```

每个训练结果导出 `protocol_audit.json`、`source_manifest.json`、`splits.json`、词表、训练日志、验证/测试原始评分和患者指标；模型还导出validation选择清单。源码哈希覆盖新模块与复用依赖。`direct_scores`是同检查点评分回放：只有冻结旧direct的calibration场景可以称冻结direct参照，joint模型的回放不能冒充另行训练的direct模型。

## 验证与代码审查

新增测试覆盖正例与旧policy一致、未知候选屏蔽、OOV/空目标审计、train-only控制、已发表损失公式、无固定点置换、静态gate、家族内排序不变量、梯度、检查点冻结、协议不匹配拒绝、原入口兼容，以及真实NIfTI读取→训练→导出的闭环。

审查重点：没有改旧默认值；输出不覆盖；验证不接触test选择；复用检查点先核验再创建输出；候选协议变更不混作同任务；聚合不静默删掉未匹配运行。测试统计与最终审查记录见 `PAPER2_REVISION_CODE_REVIEW.md`。

独立医院/队列的性能验证仍需要相应数据，不会用合成夹具或同源proxy代替。冻结外部测试命令如下；患者ID必须经核查为全局唯一，相同患者被重新命名的重复病例无法仅靠代码发现。标签命名和分割协议也需核实可比。

```bash
python -m glioma.cli.evaluate_paper2_transfer \
  --source_run /path/to/completed_revision_run \
  --data_root /path/to/independent_cohort --metadata_tsv /path/to/metadata.tsv \
  --out_dir /path/to/new_external_evaluation --segmentation_source 'actual mask provenance'
```

外部输出名为 `external_records.json`，与内部OOF汇总入口隔离。当前完成外部测试能力，外部泛化与官方完整医学图文系统复现均未产生真实数据结果。

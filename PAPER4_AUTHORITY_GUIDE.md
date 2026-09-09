# ACF-SPD：新 authority 模型与干预实验运行说明

本实现独立于 MRI 训练入口。研究定位是“理论与合成机制验证 + 既有 MRI 语义融合实证基础”。旧 81 次实验保持原协议归属，不计入新 authority 实验，也不被表述为临床 authority 已实现或已验证。

正式训练由服务器执行。已准备 60 次开发学习率试验、300 次主确认训练、210 次归因训练；rule-only 无训练。开发学习率试验不包含在 510 次正式训练中。

## 1. 服务器快速开始

解压交付的 ZIP 后保持 `glioma/` 文件夹名。建议新建 Python 3.11 或 3.12 环境，不覆盖原 MRI 环境。安装与服务器驱动匹配的 PyTorch 2.8.0 构建，然后安装其余依赖。例如已有可用 PyTorch 环境时：

```bash
python -m pip install -r glioma/requirements-authority-server.txt
bash glioma/authority_server.sh check
```

PyTorch 的 CUDA 构建应按服务器驱动选择；本机开发使用 PyTorch 2.8.0+cu126。安装方式见 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)。通用投影采用 [CVXPYlayers](https://github.com/cvxpy/cvxpylayers)，本包固定 cvxpy 1.6.5 / cvxpylayers 0.1.9。该可选依赖链要求较新的 Python；原 Python 3.9 可运行原生解析 S1/S2 路径，但不能视作完成通用层验收。

单卡按下面顺序执行。`/your/output/authority_v1` 换成服务器输出位置；所有阶段必须使用同一个正式输出根目录：

```bash
bash glioma/authority_server.sh plan --root /your/output/authority_v1
bash glioma/authority_server.sh development --root /your/output/authority_v1 --device cuda
bash glioma/authority_server.sh freeze --root /your/output/authority_v1
bash glioma/authority_server.sh main --root /your/output/authority_v1 --device cuda
bash glioma/authority_server.sh attribution --root /your/output/authority_v1 --device cuda
bash glioma/authority_server.sh rule-only --root /your/output/authority_v1 --device cuda
bash glioma/authority_server.sh aggregate --root /your/output/authority_v1
```

运行前也可用独立目录检查 GPU 完整流程：

```bash
bash glioma/authority_server.sh smoke --root /your/output/authority_smoke --device cuda
bash glioma/authority_server.sh aggregate --root /your/output/authority_smoke --phases smoke
```

smoke 固定为 S1/S2 × 四个核心配置，训练/验证/测试 = 1024/256/256、8 epochs、数据种子 41、模型种子 100、学习率 0.001。它只检查工程流程，不冻结正式学习率，不用于论文优势结论。

`AUTHORITY_PYTHON=/path/to/python` 可指定解释器。也可直接运行 `python -m glioma.cli.run_authority_protocol ...`，此时将 `glioma` 的父目录加入 PYTHONPATH。服务器脚本会自动设置该路径。

## 2. 调度、恢复与多 GPU

每个任务保存 best.pt、last.pt、history.json、config.json 和 DONE.json。最优检查点按验证 Brier 选择。恢复使用上一完整 epoch 的模型、优化器与随机数状态；未完成的 epoch 重放。相同代码和配置下，已完成任务会跳过。测试验证过恢复结果与不中断训练逐参数一致。

`--limit 2` 每次运行本分片尚未完成的两个任务；再次运行继续推进。默认遇到错误停止；`--continue-on-error` 会记录失败并继续，最终仍以非零状态退出。

每个运行有独占 RUNNING.lock。若进程被强杀或服务器断电，先确认锁中进程已不存在，再删除该任务自己的旧锁后恢复。不要同时使用重叠分片。

例如两张 GPU 分别在两个终端运行：

```bash
CUDA_VISIBLE_DEVICES=0 bash glioma/authority_server.sh development --root /your/output/authority_v1 --shards 2 --shard 0
CUDA_VISIBLE_DEVICES=1 bash glioma/authority_server.sh development --root /your/output/authority_v1 --shards 2 --shard 1
```

两边开发阶段全部结束后执行一次 freeze，再以相同方式分别执行 main 和 attribution。每个 phase 的分片按稳定任务序号划分；rule-only 也支持分片。CPU 线程默认 4，可用 `--threads` 调整。请在每个阶段完成后再进入下个阶段。

开发扫描 10 种配置 × 两任务 × 三种学习率 {0.001, 0.0003, 0.0001}，数据种子 41、模型种子 100，所有候选均按完整预算训练。只读取开发验证 Brier，完全不读取正式测试结果。freeze 要求 60 个任务全部完成，检查预算及代码指纹，输出 frozen_protocol.json 和两个正式 manifest。任何缺失或代码变化都会阻止冻结或正式运行。

原始计划 manifest 中学习率为 null；这表示尚未选择，不能直接训练。冻结之后每个 task/variant 使用其自身开发最优学习率；归因配置沿用对应主模型学习率，不额外调参。

## 3. 实现机制与信息边界

- `AuthorityBatch` 分开保存 label-free 的 `AuthorityEvidence` 与监督标签。规则编译器只接受 Evidence，不能接收 Batch/标签。
- 本版注册 S1/S2 规则语法。authority specification 由 task、target_id、每关系的 authority_sources、结构化 statements、独立可用性/有效性、rule_version 组成。来源是 tokens 源轴上固定的 0–3；角色逐样本随机。每条普通关系的授权来源集合在这两个任务中为单元素集合，不是永久模态等级。
- 编译器输出逐关系依赖矩阵、授权来源、激活关系、B/b、原子约束分组与状态。预定义不可比冲突探针添加 p[0] >= 0.8 和 p[0] <= 0.2 两条不相容约束。
- SPD 维度 8，隐藏维度 64，两层图更新，跨节点通信总质量 0.35。源内线性适配后的 token 构成均值增强的协方差块矩阵，显式保留一阶信息；加入对角稳定项后做迹归一化。
- 图状态使用 SPD 对数的对称向量坐标。允许邻域中的加权聚合后做逐节点 LayerNorm/非线性更新。对称坐标经矩阵指数对应 SPD 状态；计算中保留对数坐标，避免反复计算等价的 log(exp(.))。不采用纯线性传播加全局均值池化，读出保留各源位置。
- 权威状态的入边仅来自该关系授权自身和固定支持节点；辅助可以读取权威。attention 仅在允许邻域归一化，归一化不使用其他节点统计，源内上下文不读取其他源观测。
- 四个固定概念模板对所有方法相同；适配器可训练，但单次前向中支持节点保持原始状态、不能接收患者辅助信息后回写。模板不具临床含义。
- 所有主要基线的预测头获得相同目标、policy、来源可用性、独立有效性、质量、结构化 statements 与编译 B/b。该全局信息只进入预测头；不反向作为当前前向的保护状态输入。跨训练步骤的参数学习不属于固定检查点的非干扰保证。

解析输出层实现受限 simplex 欧氏投影与不相交成对排序的 box 欧氏投影。S1 softmax，S2 sigmoid。完整模型、普通 graph 联合投影、Transformer 联合投影均用 `Brier(p*, y) + 0.1 mean((p*-p0)^2)`。Brier 统一按输出维度取均值。soft 配置加 0.1 平均平方正关系残差，不加投影；其余非联合模型使用 Brier(p0,y)。不叠加新的对齐或拓扑损失。

`GeneralAuthorityProjection` 支持任意输入 B/b 的双精度、可微 CVXPYlayers 投影；先独立检查可行性，再校验数值残差。`AuthorityFusion(projection_backend="general")` 可用通用层核验已编译约束；正式 S1/S2 默认解析后端。扩展任意新任务的规则语法需要新编译规则；通用优化器本身不推断 authority。

状态码：0 constrained；1 unconstrained；2 conflict；3 no_input；4 solver_failure。无效来源退出融合、撤销关系；unknown 保留为辅助输入但不激活硬关系；缺失撤销关系；仍有输入但无关系时返回无约束预测；冲突或全缺失拒答。数值求解失败独立记录。拒答行保留原始分数用于审计，不能把这些分数当作已返回预测。

## 4. 合成任务与干预

S0 枚举独立公平 X1/X2/A 的 8 种情况，Y=X_A；带 policy 错误率 0，无 policy 的 Bayes 错误率 0.25。

S1 使用原始粗位 a、细位 b、外生目标 q。q=0 时 Y=2a+b；q=1 时 Y=2(1-a)+b。授权观测提供原始 a，编译器依据 q 限制对应候选类别；辅助源提供含噪 b。一个辅助源含细位信号，其质量为 r_b，其他非权威源提供干扰。target_switch 只切换 q 与相应标签，不改变原始 tokens/statements。来源角色逐样本随机。

S2 为 6 维标签，关系为两对分量的概率大小顺序，其余两维依靠辅助连续证据。q 改变两关系的合资格来源。先生成符合顺序关系的潜在概率，再用共享均匀随机数采样标签；随机标签本身允许违反概率顺序。target_switch 保留原始证据，用改变后的外生定义生成对应潜在概率/标签。

每个 task/split/data seed 使用固定独立命名随机流；修改干预参数不重采潜在样本，低样本训练集是完整训练集的精确前缀。不同 split 使用不同随机流。

S1 保存 clean + 10 个专门干预 + 27 个 eta/r_b/kappa 单元 + 4 个冲突强度单元，共 42 个事件集合。S2 保存 clean + 10 个专门干预 + 3 个可靠性单元，共 14 个事件集合。

- eta={0,.05,.15} 为被接纳陈述错误率；r_b={.6,.8,.95}；kappa={0,.5,1}；这些单元均保留真实标签。
- 额外强度 {1,2,4,8} 固定 eta=0、kappa=1、r_b=.8，只改辅助粗位。
- invalid/unknown/missing 固定原始观测、目标和标签，只改有效性或可用性。all_missing 全部撤销。
- wrong_statement 翻转被接纳陈述；wrong_policy 将来源位置循环移位；shuffled_policy 在同一潜在样本集内按固定顺序置换 policy。真实标签不变。
- target_switch 是改变外生任务定义的独立实验；不能与固定标签干预混合解释。
- conflict 为预定义不可相容约束；bypass 为故意加上辅助参与的共享 sigmoid gate 的正对照。同一旁路结构下比较修改辅助前/后的状态，避免只检测到“换了网络”。Transformer 不使用该 graph 旁路，其依赖审计仍以普通输入扰动为准。

S2 的 r_b 只改变两个未被授权的辅助源的连续信号强度/质量，保持两个被接纳来源的 tokens、质量与陈述完全不变；它不解释为每个 Bernoulli 标签的精确正确率。错误 authority 压力测试单元不适用“正确规则下”的性能保证。

## 5. 对照与训练预算

| 配置名 | 含义 |
|---|---|
| rule_only | 无训练，均匀初始概率 + 相同关系投影 |
| uniform | 相同 policy 的均匀通信图 |
| learned | 相同 policy 的学习图 |
| reliability | 同一学习图的邻居权重加入 log(quality) 可靠性 gate |
| dominant | 在训练集拟合等预算来源 ridge probes，用验证 Brier 选择统计优势来源，保护它 |
| transformer | 同等信息的 Transformer |
| transformer_joint | Transformer + 相同联合投影与损失 |
| learned_joint | 普通学习 graph + 相同联合投影与损失 |
| graph_only | authority graph，输出不投影 |
| soft | 学习 graph + 软关系惩罚 |
| acf | authority graph + 联合关系投影 |

learned、transformer、graph_only 的相同检查点另报告测试时投影，复用 p*，无新增训练。rule_only 每个任务/数据种子评估一次，不虚构三个模型种子。

正式默认训练/验证/测试=12000/3000/6000，50 epochs，batch 256，AdamW，weight decay 1e-4。学习率按开发集冻结。模型种子 101–103，数据种子 42–46。

主确认：10 × 2 × 5 × 3 = 300。归因围绕 acf 与 learned_joint：Euclidean 60、去锚点 60、低样本 2048 共 60，另 acf 角色置换 mask 30，总计 210。

Euclidean 消融保留同一均值增强矩阵及维度，直接使用对称矩阵坐标，去掉矩阵对数几何。主 mask 对照 acf/graph_only/learned_joint 的网络与参数量完全相同；角色置换 mask 用同一网络，同时置换源节点的行列，保留边数与整体入/出度分布。记录逐节点度数，不能声称逐节点度数不变。缺失后的再次裁剪可能改变度数分布，需按相应事件单独解读。其他架构参数量单独记录，不宣称跨 Transformer 参数完全相同。

## 6. 输出、统计与论文结论

`phase/job/events/*.npz` 为可用 NumPy `allow_pickle=False` 读取的逐样本记录。包含 sample_ids、event_ids、干预前后 p0/p*、实际输出概率、激活关系、B/b、原子约束、状态、保护状态、逐关系变化、逐节点度数、有效性/可用性/质量、目标及前后标签；旁边 JSON 保存种子、规则版本、干预参数、代码/配置标识和平均耗时。NPZ 中前后对应行拥有相同潜在样本。

无关系样本的 AVR 分母为零并记为 null/N/A；冲突、全缺失、数值失败不进入已返回预测的 AVR 或 Brier 分母。必须同时报告预测覆盖率。固定覆盖率风险按所有输入为基准，只有可用预测达到指定覆盖时才计算；不能把 50% 可用预测内的 80% 宣称为总体 80% 覆盖。

保护状态比较限定两边目标、授权来源、关系方向相同、关系均激活且均可返回预测的情况。只改辅助的严格检查另要求授权输入本身不变。错误陈述、来源撤销和目标切换的变化不是保护失败；原始数组仍完整保存供检查。

汇总输出 all_runs.csv（逐运行/事件）、data_seed_means.csv（先平均模型种子）、all_paired_runs.csv（15 次配对）、paired_data_seed_means.csv（5 个独立数据集）、paired_intervals.csv（数据种子层面 95% percentile bootstrap）和 RESULTS.md。单个检查点的 6000 个干预事件不作为独立实验。主确认和归因分别标记 phase；不合并成 510 个独立同质样本。

汇总默认要求正式训练完整；中途查看需显式加 `--allow-incomplete`，缺失清单仍保留。不同代码指纹拒绝混合。训练耗时指各 epoch 的训练计算总时间；推理耗时包含编译、前向、投影和结果转 CPU，文件压缩不计入推理时间；另保留 GPU 峰值分配显存。

结论必须基于信息匹配的配对对照。只有完整模型优于普通 graph + 相同联合投影，才支持 graph 的增量价值；若只是保护状态不同而任务表现接近，应限定贡献为可验证的通路约束。Transformer + 相同信息/投影表现相当时，承认架构优势未成立。零违反率单独只说明约束实现正确。错误 authority 的真实任务代价完整保留。在新增患者级多源证据前，不声称真实临床 authority 验证。

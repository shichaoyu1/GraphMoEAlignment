# Idea 3 连续偏好实验运行说明

本轮新增独立的 token 数据模块、MRI 读出器、选择策略、训练器、受控 oracle、汇总和分级入口。默认输出根目录为当前仓库的 output/idea3_pareto_v1。旧训练入口可继续按其原配置使用。

## 可启动的实验

| 模式 | 功能 |
|---|---|
| controlled | 一个受控任务、一个训练法 |
| controlled-grid | 三个受控环境与九种核心训练配置 |
| prepare / prepare-grid | 全视野数据准备及同折共同编码器训练 |
| train | 单折单种子单方法训练 |
| ablation | 九种核心配置依次训练 |
| reference | 五个固定偏好参考策略 |
| matrix | 多折多种子的九配置矩阵 |
| authority | 选定策略的连续偏好分级、MRI 交换及报告撤销 |
| analyze | OOF 曲线、超体积配对区间和 PDF |

方法名为 direct_sft、direct_pg、dense_sft、moe_sft、dense_pg、moe_pg、dense_nft、moe_nft、moe_linear。另有 dense_epsilon，可通过 CLI 的 --epsilon 指定表型约束，阈值必须根据训练参考风险确定。

## AutoDL 启动

在仓库目录设置共享环境变量。authority 路径对应本次已经上传的归档。数据和元数据路径以实例实际位置为准。

```bash
cd /root/autodl-tmp/repo/glioma
export IDEA3_AUTHORITY="$PWD/output/idea3_rule_mri_swap/utsw_authority_grade"
export IDEA3_DATA_ROOT=/root/autodl-tmp/dataset/UTSW-Glioma
export IDEA3_METADATA=/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv
export IDEA3_PARETO_OUTPUT="$PWD/output/idea3_pareto_v1"
mkdir -p "$IDEA3_PARETO_OUTPUT/logs"
```

先独立启动受控短程，验证生成器和导出。新建的 smoke 输出根与正式运行分开。

```bash
IDEA3_PARETO_OUTPUT="$PWD/output/idea3_pareto_smoke" \
nohup bash scripts/run_idea3_pareto_autodl.sh controlled-grid \
  --warmup 10 --updates 10 --local-steps 4 --proposals 16 \
  --validate-every 10 --eval-repeats 4 --test-repeats 4 \
  > "$IDEA3_PARETO_OUTPUT/logs/controlled_smoke.log" 2>&1 < /dev/null &
echo "PID=$!"
```

训练首折共同 MRI 编码器，默认 30 epochs。如需短程检查，使用独立输出根并追加 --epochs 1。

```bash
nohup bash scripts/run_idea3_pareto_autodl.sh prepare \
  > "$IDEA3_PARETO_OUTPUT/logs/prepare_fold0.log" 2>&1 < /dev/null &
echo "PID=$!"
```

准备完成后，单独启动主方法或消融。它们共用已经冻结的 tokens.pt。

```bash
nohup bash scripts/run_idea3_pareto_autodl.sh train \
  > "$IDEA3_PARETO_OUTPUT/logs/main_fold0.log" 2>&1 < /dev/null &
echo "PID=$!"
```

```bash
nohup bash scripts/run_idea3_pareto_autodl.sh ablation \
  > "$IDEA3_PARETO_OUTPUT/logs/ablation_fold0.log" 2>&1 < /dev/null &
echo "PID=$!"
```

主方法默认 moe_pg。通过 IDEA3_METHOD、IDEA3_FOLD、IDEA3_SEED 选择单项实验。相同配置已完成的任务会跳过，正在运行的同一目录由锁保护。中断训练从 last.pt 恢复，源码或配置变化要求另选输出目录。

完整矩阵先执行 prepare-grid，完成后执行 matrix。通过 IDEA3_SEEDS="42 43 44" 和 IDEA3_FOLDS="0 1 2 3 4" 指定范围。多个模式的依赖用顺序命令连接，避免训练读取尚未完成的缓存。

```bash
nohup bash -c 'bash scripts/run_idea3_pareto_autodl.sh prepare-grid && bash scripts/run_idea3_pareto_autodl.sh matrix' \
  > "$IDEA3_PARETO_OUTPUT/logs/matrix.log" 2>&1 < /dev/null &
echo "PID=$!"
```

分级机制扩充在所选 fold/seed/method 的主策略完成后运行。

```bash
nohup bash scripts/run_idea3_pareto_autodl.sh authority \
  > "$IDEA3_PARETO_OUTPUT/logs/authority_fold0.log" 2>&1 < /dev/null &
echo "PID=$!"
```

分级入口会从原始 MRI 计算对应同折新编码器的全视野特征，训练三个公共分级头。该步骤的计算与主 selector 训练分别计量。输出逐患者、偏好、随机数和供体概率。

```bash
bash scripts/run_idea3_pareto_autodl.sh analyze --bootstrap 10000
```

## 当前实现与完整论文协议

4000 步监督验证后，可直接复用已经训练好的监督权重。controlled 和 train 支持 --init-policy 指向 best.pt，配合 --warmup 0 跳过重复预热。加载时核对折、种子、证据预算、生成步数、采样噪声与模型形状，MRI 训练还核对冻结 token 文件哈希。初始权重会先接受验证并作为更新 0 的候选检查点，后训练退步时能够保留原权重。预训练成本与分叉成本分别报告。

示范策略默认为 top8，保留此前实验定义。--teacher-mode best 改为仅在候选库里最优得分 1e-6 容差内采样，解决最优候选不足 8 个时混入次优示范的问题。这改变了监督目标，需要使用新输出目录并标注新协议。它仅保证在当前候选库内最优，不保证全局前沿覆盖。

可用 bash scripts/run_idea3_pareto_forks_autodl.sh 一次顺序运行共同最优和传感器环境的四种分叉，共八项任务。它们各自复用相同 MoE 监督初始化，比较继续 top8 监督、best 示范监督、PG 和 NFT。默认从 output/idea3_pareto_pretrain 查找已有权重，新结果保存在 output/idea3_pareto_mature_forks。可通过 IDEA3_PRETRAIN_ROOT 和 IDEA3_PARETO_OUTPUT 覆盖。此比较用于区分示范质量和后训练方法，完成前不预设 RL 或 MoE 获益。

2026-11-25 的服务器学习验证发现，PG 对所有时间步取均值，额外把终端奖励的轨迹梯度缩小到 1/T。现已改为时间步求和、轨迹批次取均值。解析线性终端奖励的真梯度为 1，旧实现在 T=16 时估计约 0.0618，修正后的估计约 0.9890。该修正改变 PG 与参考 KL 的相对权重，旧 PG 结果保留为修正前先导。

新 history 会记录裁剪前梯度范数、独立块风险、标量化导数、组内得分标准差和参考 KL。更新后使用新的输出目录，避免把不同梯度定义的检查点混合恢复。生成器的监督预热是否充分仍需用同架构 moe_sft 与 moe_pg 对照检验，梯度修正本身不能保证复杂模型收益。

首版已实现三种已知风险环境、九种核心训练配置、固定偏好参考、可指定阈值的 epsilon 策略、共同读出、主输出、分级干预和统计入口。数值实验用于检验执行和梯度，短程结果不能证明模型收益。

direct_sft 当前使用固定候选示范的监督选择器，尚未加入直通 top-K 的任务梯度分支。NFT 使用单次在线更新的 behavior snapshot，并以冻结监督模型作为 reference。上游逐目标 NFT 损失按 64 维选择变量改编，医学迁移的随机核与 SD3 原生 dpm2 不同。

汇总支持配对 HV 区间及四项预定比较的 Holm 校正。缺失类别的 bootstrap 比例显式记录。原测试类别权重固定的敏感性、epsilon 三阈值自动调度、路由回放、真正留空偏好区间、预算扫描与全部内容侧信道诊断尚未集成为一键矩阵。它们仍属于完整论文协议的后续实现项，当前训练矩阵的结果按实际完成配置解释。

受控外部规则的结构检查由测试覆盖，完整 101 偏好状态网格和 MC 梯度收敛试验仍需单独执行。正式五折三种子与 618 人分级尚待 AutoDL 验证。

参考实现来自 Apache-2.0 的 ParetoSlider T2I，固定提交 e56e5024a3e996944119ef7cb910d314900b3fc1。移植的 NFT 数学结构对应其 train_pareto_nft_sd3.py，数据、网络、采样与医学输出为本项目独立实现。

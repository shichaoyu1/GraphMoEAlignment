# MRI 基础诊断与可选后训练对照

本阶段承接 fold0/seed42 的 MRI 先导结果：连续偏好尚未产生方向正确的权衡，NFT 相对监督模型的 HV 差值区间跨零。默认只做单种子五折的简单参考诊断，不启动完整模型矩阵，也不自动启动 NFT。

## 默认基础诊断

入口为 `scripts/run_idea3_pareto_mri_screen_autodl.sh screen`。默认 seed42、fold0–4；已完成的 fold0 冻结 token 直接读取，不调用旧输出目录的 prepare。其余折仅训练同协议的共同读出器，默认 30 epochs，并缓存 token。每折包含：

- 仅在训练标签上搜索全体患者共用的 8-token 选区，固定 w=0、0.25、0.5、0.75、1。候选为 256 个随机选区，每个权重 128 次局部替换，目标按训练折尺度归一化后线性加权。
- 随机 8-token 参考，每患者 256 次；全部 64-token 读出器参考。
- 训练折分子频率和分子常数 0.5，均配训练折表型中位数。

随机选区与标签独立；固定选区只在训练集选择。验证和测试均使用冻结选区评价，不按其真值选候选。全64 token 属于不同证据预算，单列为读出器参考。

逐患者 CSV 保留分子字段真值、概率、每次随机选区与表型误差。跨折从逐患者平方误差重算全体 OOF 各字段正负类别等权的分子 Brier；表型 MAE 按患者等权，以百分点报告。患者在折内配对 bootstrap 10,000 次，重算类别分母，显式报告无定义重采样次数。该口径与旧分析器的折等权汇总分开报告。

主要观察 w=1 减 w=0 的分子 Brier 与表型 MAE 差值、区间及每折方向；预期为分子差值负、表型差值正。汇总给出方向复现折数和区间是否支持该方向，但始终标记 `diagnostic_only`。有限选区搜索不是 oracle，未见取舍不能证明真实前沿不存在。

验证集作为预算诊断的主要参考，但读出器已用该验证集选检查点，仍是探索性结果。fold0 测试结果也已看过。本轮不构成独立确认，不得据此对测试集再选掩膜、权重或检查点。旧 AP/表型先导门槛及停止决定不变。

## AutoDL 后台命令

```bash
cd /root/autodl-tmp/repo/glioma &&
git pull --ff-only origin main &&
(
  run="$PWD/output/idea3_pareto_mri_screen"
  mkdir -p "$run/logs"
  IDEA3_PARETO_OUTPUT="$run" \
    nohup bash scripts/run_idea3_pareto_mri_screen_autodl.sh screen \
    > "$run/logs/launcher.log" 2>&1 < /dev/null &
  echo "PID=$! LOG=$run/logs/launcher.log"
)
```

默认读取已上传的权威归档 `output/idea3_rule_mri_swap/utsw_authority_grade`；旧 token 来源 `output/idea3_pareto_mri_pilot_poolfix/prepared/seed42/fold0/tokens.pt`。影像缓存优先使用上述 token 来源根的 `prepared_images/experiment1`，再检查原失败运行 `output/idea3_pareto_mri_pilot/prepared_images/experiment1`；均无缓存时，从原始 MRI 重新准备到新输出根。需要重新准备时，默认 MRI 目录 `/root/autodl-tmp/dataset/UTSW-Glioma`，元数据 `/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv`。

可用环境变量指定 `IDEA3_MRI_PILOT_ROOT`、`IDEA3_PREPARED`、`IDEA3_AUTHORITY`、`IDEA3_DATA_ROOT`、`IDEA3_METADATA`、`IDEA3_DEVICE`、`IDEA3_SEED`、`IDEA3_FOLDS`。归档内容、输入哈希与检查点签名会核对，不通过修改旧 config 绕过检查。改变预算、数据或实现版本时使用新的输出根。

`IDEA3_DRY_RUN=1` 仅打印调度命令并检查基本路径，不启动预处理或训练；不会验证张量文件内容。数值预算分别为 `IDEA3_READER_EPOCHS`、`IDEA3_PROPOSALS`、`IDEA3_LOCAL_STEPS`、`IDEA3_RANDOM_REPEATS`、`IDEA3_BOOTSTRAP`。

所有新输出和日志位于同一根：

```text
output/idea3_pareto_mri_screen/
  logs/launcher.log
  prepared/seed42/fold1/...        # 新训练的共同读出器与冻结 token
  screen/seed42/fold0/...          # 每折选区、预测和摘要
  analysis/seed42/summary.json     # 跨折主摘要，仍是探索性诊断
  analysis/seed42/risk_curves.csv
```

如果原缓存已删且原始数据路径也不存在，脚本会提示缺失输入；应设置正确的缓存或原始数据路径。不要将代码仓库复制到 output，也不创建 import_root。

## 可选的相同初始化后训练比较

单独入口 `posttrain`，默认只跑 fold0、seed42。它不会由 screen 自动调用。每个 Dense/MoE 家族均从同一已有 SFT 检查点分叉：继续 SFT、PG、NFT，各 200 步、每 50 步验证一次、相同推理采样预算、均保留 update0 候选，优化器重置。原监督预训练成本与新增分叉成本分列。这里的“相同预算”指更新步数相同，实际运行时间也必须报告。

新 NFT 必须显式使用 `--nft-objective population_stch`：与 PG 共享独立总体风险估计的平滑 Tchebycheff 梯度系数、组内留一标量优势，再按整个批次同一 RMS 标准化。该路径避免逐目标或逐患者标准化改变目标相对权重，记录优势尺度、标准差和裁剪比例。NFT 仍是有界重建代理损失，并非与 PG 相同的梯度估计。原 `legacy` 公式及默认行为保留，旧受控与 MRI 结果不改写。

```bash
# 先审阅基础诊断；这是独立的可选运行。
run="$PWD/output/idea3_pareto_mri_screen"
mkdir -p "$run/logs"
IDEA3_PARETO_OUTPUT="$run" \
  nohup bash scripts/run_idea3_pareto_mri_screen_autodl.sh posttrain \
  > "$run/logs/posttrain.log" 2>&1 < /dev/null &
```

主要比较为各家族 NFT 减继续 SFT 的配对 HV 差值，汇总新增 `nft_minus_sft`；`pg_minus_sft` 为参考。原 MoE-PG 中心的结果字段保留。这些区间未经多重比较校正，只以当前拟合模型为条件。旧初始化与新分叉分别在旧根和新根，不在同一目录混入 legacy/aligned NFT；改变目标应新建输出根。

## 下一阶段决定

只有方向能跨折复现、且简单同预算参考留下稳定差距，才考虑扩大模型比较；优先单种子检查其他折，再扩训练种子。若简单方案可共同改善两项任务，接受共同优化结论。当前筛查不会自动授予“全量通过”。此前58人权威规则内 MRI 交换证据继续单独保留，本阶段不会重跑分级入口。

## 实现验收范围

本地检查覆盖验证/测试标签改变不影响训练选区、归档和输入哈希、恢复与损坏拒绝、跨折患者分母和重复拒绝、OOF 类别平衡、缺失类别区间，以及脚本调度和旧入口回归。使用真实 fold0 缓存做过 CUDA 小预算筛查（8 候选、2 次局部替换、4 次随机采样、200 次 bootstrap），新 NFT 也做过 CUDA 短程；这些仅验证执行，不用于研究结论。默认 legacy NFT、PG、SFT 另与修改前代码逐位比较，历史和参数一致。完整五折诊断仍需在服务器运行。

# Idea4.5 v3 代码与服务器运行说明

当前代码实现 [v3 协议](IDEA4_5_PROTOCOL_V3.md) 的 E0–E2：测量检查、现象确认和机制定位。E3/E4 的控制器实验仍受协议放行条件约束；只有试点和确认结果表明四分支诊断能预测有限步后果，才进入控制器比较。

## 与旧 Idea4 的边界

旧 Idea4 研究 SPD 几何和图通信是否改善预测。Idea4.5 研究来自一种模态可用性上下文的更新，如何改变另一上下文的风险，以及指定计算路径的反向作用是否随上下文改变。代码仍复用 Paper4 骨干，但数据、审计、运行和汇总入口均位于 `idea4_5` 模块。

## v3 已落实的设计

- 合成数据含 `exact_copy`、`repeat_measurement`、`novel_signal`、`null_signal` 四种条件。条件共享标签、对象 ID、划分和基础随机流。
- data、split、init、order、mask、audit seed 独立记录；旧 `model_seed` 和 `extra_information` 作为兼容别名。
- 合成数据按总样本的 60/10/10/10/10 分配 fit/select/measure/val/test。真实 NPZ 按 group 隔离 train/val/test，并在训练集内按组划分角色。
- rich/sparse 训练上下文按批次数精确配额，所有运行记录实际对象曝光与批次曝光。
- 每个 10%/50%/90% 检查点默认抽取 20 个独立源批次；不使用刚训练过的末批。
- 同一快照执行 mechanism SGD 和保留训练状态的 stateful 审计。输出四分支、反向边、两上下文等位移学习率、head/encoder 参数控制、AdamW 零梯度背景。
- 保存逐对象风险、完整模型/优化器/RNG 快照、源对象 ID、测量对象 ID、块级梯度与参数位移。
- 汇总先在每个 data seed 内平均 train seed，再以 data seed 为独立单位 bootstrap。
- `edge_normalization=fixed` 固定公共边系数，用于检查节点数量引起的动态归一化解释。
- UTSW 原始 NIfTI 可直接作为患者级真实数据复验；默认任务为三分类 grade，使用固定的患者级 5 折划分。

`parameter_zero` 是参数块当前梯度控制。它不能替代计算节点级的 head/encoder 因果干预，因此相关结果只作为替代解释检查。图消息的 on/off 仍保持完全相同的 forward 值。

## 真实数据 NPZ 合同

每个 `train/val/test` 必须包含：

```text
{split}_tokens       [N,G,M,T,C]
{split}_labels       [N]
{split}_sample_ids   [N]
{split}_group_ids    [N]
```

可选 `{split}_modality_mask [N,M]` 和 `{split}_token_mask [N,G,M,T]`。加载器会拒绝跨 split 的 sample/group 重叠。当前配对协议要求 rich 的全部模态完整观测且每模态至少两个有效 token。

`--label_mode sign` 实现 `<0` 对 `>=0`；`--label_mode sign_nozero` 实现 `<0` 对 `>0` 并删除零分对象。二者不能在一张结果表混用。

## UTSW 原始数据

服务器入口默认读取以下两个路径：

```text
/root/autodl-tmp/dataset/UTSW-Glioma
/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv
```

加载器逐患者读取四种 MRI（T1、T1CE、T2、FLAIR）和肿瘤分割，从肿瘤区域裁取 7 个切片。原始 NIfTI 只读，不复制到输出目录。默认的 `pooled` 编码把每个模态确定性池化为 8×8 token，因此 UTSW 结果应表述为特征级可用性干扰证据。CLI 另提供 `--utsw_encoder cnn`，但它不属于默认确认配置。

`UTSW_TASK` 支持 `grade`、`idh`、`mgmt`、`1p19q`。默认 `grade` 将 II/III/IV 映射为 0/1/2；缺失目标标签的患者会被排除。每折以一个患者折作测试、下一折作验证，其余折作训练，训练集内再按患者隔离 fit/select/measure。

## 服务器入口

从包含 `glioma/` 的工作区根目录执行。测试模式运行单元测试和一个小型 CPU smoke run：

```bash
bash glioma/run_server_idea4_5.sh test
```

试点严格对应 3 data seed × 2 train seed × 4 单元，共 24 条轨迹：

```bash
bash glioma/start_server_idea4_5.sh pilot
```

全量主单元默认 20 data seed × 2 train seed × 4 单元 × 2 边归一化，共 320 条轨迹；另加入 identity/nodewise 和 uniform/pooled 两个架构控制，共 80 条，总计 400 条。设置 `RUN_ARCH_CONTROLS=0` 可只跑主单元。20 个 data seed 是预算配置；确认样本量仍应根据试点方差冻结。后台启动：

```bash
bash glioma/start_server_idea4_5.sh full
```

常用资源覆盖示例：

```bash
PYTHON_BIN=/opt/conda/envs/glioma/bin/python \
OUTPUT_ROOT=/data/idea4_5_v3/full \
NUM_SAMPLES=6000 EPOCHS=30 AUDIT_SOURCE_BATCHES=20 \
bash glioma/start_server_idea4_5.sh full
```

公共三模态 NPZ：

```bash
DATA_PATH=/data/mosei_v3.npz LABEL_MODE=sign \
OUTPUT_ROOT=/data/idea4_5_v3/mosei \
bash glioma/start_server_idea4_5.sh public
```

先用实际 UTSW 数据执行一折短测试：

```bash
bash glioma/start_server_idea4_5.sh utsw-test
```

测试通过后启动 5 折全量。默认每折 2 个训练种子、2 个 rich 比例、2 种边归一化，并加入两个架构控制，共 60 条轨迹：

```bash
bash glioma/start_server_idea4_5.sh utsw
```

路径已是默认值，也可以显式指定输出位置：

```bash
DATA_ROOT=/root/autodl-tmp/dataset/UTSW-Glioma \
METADATA_TSV=/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv \
UTSW_TASK=grade OUTPUT_ROOT=/root/autodl-tmp/idea4_5_v3/utsw \
bash glioma/start_server_idea4_5.sh utsw
```

UTSW 默认每个检查点抽取 5 个源批次，先控制真实图像审计的计算量。服务器短测试确认显存和耗时后，确认性运行使用：

```bash
UTSW_AUDIT_SOURCE_BATCHES=20 \
OUTPUT_ROOT=/root/autodl-tmp/idea4_5_v3/utsw_confirm \
bash glioma/start_server_idea4_5.sh utsw
```

启动器会显示 PID、日志和 PID 文件路径。查看进度：

```bash
tail -f glioma/output/idea4_5_v3/logs/full_YYYYMMDD_HHMMSS.log
```

已有 `run_summary.json` 的轨迹自动跳过，因此同一 `OUTPUT_ROOT` 可安全重启。失败轨迹写入 `failed_runs.tsv`，成功轨迹仍会汇总。

## 输出

每条轨迹包含：

- `config.json`、`data_roles.json`：完整配置、角色索引及稳定 ID。
- `checkpoint_epoch_*.pt`：模型、优化器、Python/NumPy/PyTorch/CUDA RNG 状态。
- `history.json`：训练风险、两上下文对象/批次曝光、裁剪触发和验证结果。
- `audits.json`：每个源批次的四分支估计、控制项及块级诊断。
- `audit_objects_*.npz`：逐对象四分支风险及源/测量 ID。
- `test_metrics.json`、`run_summary.json`、`run_status.json`：最终风险、运行摘要和状态。

全局 `aggregate/aggregation.csv` 和 `aggregation.json` 给出聚合结果及 95% bootstrap 区间；合成数据以 data seed 为聚合单位，UTSW 折间区间只描述同一队列的患者 CV 折变化，不能写成独立队列复现。`run_inventory.json` 保留全部完成和失败状态。结果解释遵守 v3 的收缩规则：`delta_rel>0` 不能写成绝对损害，`J!=0` 不能单独写成路径导致长期侵蚀。

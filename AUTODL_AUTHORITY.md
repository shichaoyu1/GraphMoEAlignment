# AutoDL：从 GitHub 同步并后台运行

适用于 AutoDL 算力市场的普通容器实例，在 JupyterLab Terminal 或 SSH 中操作。代码目录必须命名为 `glioma`，因为现有程序使用 `glioma.*` 包导入。

## 首次同步

```bash
cd /root/autodl-tmp
git clone --branch main https://github.com/shichaoyu1/GraphMoEAlignment.git glioma
cd glioma
```

已有上述克隆时，在没有运行实验的情况下更新：

```bash
cd /root/autodl-tmp/glioma
git pull --ff-only origin main
```

## 准备环境

激活已有的 Python 3.11/3.12 + CUDA PyTorch 环境，建议 PyTorch 2.8.0。不要在正在运行的实验中升级依赖或更新代码。若需要单独创建环境：

```bash
conda create -n authority python=3.12 -y
conda activate authority
# 安装与你的服务器驱动兼容的 PyTorch 2.8.0 CUDA 构建，然后执行 setup。
bash autodl_authority.sh setup
```

若已有合适环境，直接 `bash autodl_authority.sh setup`。setup 先检查 Python 与 GPU，再安装已固定版本的其他依赖；不会自动替换 PyTorch。PyTorch 安装选项见[官方说明](https://pytorch.org/get-started/locally/)。

启动器会将 OMP/MKL/OpenBLAS/NumExpr 线程数设为正整数（默认 4，可通过 `AUTHORITY_THREADS` 指定），避免容器继承的错误环境变量。setup 安装后验证依赖导入；start 在提交后台进程之前核对实际解释器、Python 版本、通用投影依赖和 GPU。若缺少 cvxpylayers，应在该解释器对应环境重新执行 setup，不应跳过通用层验收。日志中的 GPU 可用不代表其他依赖已安装。

## 一条命令后台运行全部实验

```bash
bash autodl_authority.sh start all
```

脚本保存当前解释器绝对路径，后台进程继续使用该环境。summary-only 新协议默认输出到 `/root/autodl-tmp/paper4_authority_v2_summary`，避免与旧版完整产物混跑。执行顺序：验收检查 → 60 次开发学习率试验 → 冻结学习率 → 300 次主实验 → 210 次归因实验 → rule-only → 汇总。60 次开发试验不计入正式 510 次训练。

正式协议默认 `AUTHORITY_ARTIFACT_LEVEL=summary`：每个任务只保存指标汇总，不再为每个干预保存 6000 行 before/after NPZ。完成任务默认删除 `last.pt` 和 `best.pt`；训练中断前仍保留 `last.pt` 用于恢复。默认要求输出盘至少剩余 2 GiB，适用于 15 GiB 数据盘；低于阈值会在下一个 epoch/评估前安全失败。需要审计产物时应使用独立输出目录：

```bash
AUTHORITY_ARTIFACT_LEVEL=audit AUTHORITY_CHECKPOINT_RETENTION=best \
  bash autodl_authority.sh start smoke /root/autodl-tmp/authority_audit
```

可选值为：artifact level `summary|audit|full`，checkpoint retention `none|best|all`。磁盘阈值可通过 `AUTHORITY_MIN_FREE_GB` 修改。

使用 `nohup` 脱离终端，关闭 SSH/JupyterLab 页面后继续运行；`flock` 防止同一输出目录重复启动。AutoDL 的后台与日志机制参见[官方文档](https://api.autodl.com/docs/linux/)。实例关机后计算会停止，重新开机后执行相同命令恢复，不能将后台运行理解为关机后仍运行。

查看进度：

```bash
bash autodl_authority.sh status
bash autodl_authority.sh logs
```

在日志查看界面按 Ctrl+C 只退出查看，不停止后台实验。日志持续追加到 `pipeline.log`；`pipeline.status` 标记当前阶段或失败原因。启动成功仅表示后台进程已提交，随后以 status 和日志中的 GPU/验收结果为准。

若先验证服务器 GPU 完整流程，使用单独输出目录：

```bash
bash autodl_authority.sh start smoke /root/autodl-tmp/authority_smoke
bash autodl_authority.sh logs /root/autodl-tmp/authority_smoke
```

等待 smoke 完成后再启动正式 all。也可分阶段后台运行：

```bash
bash autodl_authority.sh start development
bash autodl_authority.sh start main
bash autodl_authority.sh start attribution
```

这些命令应逐个执行、等待前一个完成；推荐直接使用 all。单独 main/attribution 要求已有冻结协议。指定 GPU 和输出目录示例：

```bash
CUDA_VISIBLE_DEVICES=0 bash autodl_authority.sh start all /root/autodl-tmp/authority_v1
```

## 恢复与结果

同一代码、同一输出目录下重复 `start all`：完成的训练跳过，未完成的训练从上一完整 epoch 恢复。阶段出错会停止，避免在不完整开发结果上继续正式训练。若发生强杀或断电，可能遗留某个训练任务的 `RUNNING.lock`；应先核对其中 PID 对应任务确已退出，再移除该任务的旧锁并恢复。后台总锁由操作系统自动释放，无须删除 `.pipeline.lock`。

运行期间不要 git pull，以免不同任务读取不同版本代码。需要改变模型/协议时应使用新的输出目录，程序会拒绝混用代码指纹。

最终汇总位于输出目录的 `summary_main_attribution_rule_only/RESULTS.md`；训练历史位于 `development/`、`main/`、`attribution/`、`rule_only/`。检查点和逐样本干预记录只在所选 retention/artifact level 要求时保留。详细协议见 [PAPER4_AUTHORITY_GUIDE.md](PAPER4_AUTHORITY_GUIDE.md)。

旧版完整输出可先做只读预览，再仅清理具有有效 `DONE.json` 且不带运行锁的任务：

```bash
bash authority_server.sh prune \
  --root /root/autodl-tmp/paper4_authority_v1 \
  --completed-only --remove-event-arrays --remove-last-checkpoints

bash authority_server.sh prune \
  --root /root/autodl-tmp/paper4_authority_v1 \
  --completed-only --remove-event-arrays --remove-last-checkpoints --apply
```

第一条命令只是 dry-run；只有第二条带 `--apply` 才会删除列出的可再生成文件。修改后的代码具有新的 source hash，旧 frozen root 不能用于混跑新的训练；正式新运行应换用新的输出根目录。

代码同步到 GitHub；实验输出、模型检查点、虚拟环境和患者数据保留在本机/服务器，不通过 Git 同步。

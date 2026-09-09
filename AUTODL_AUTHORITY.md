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

## 一条命令后台运行全部实验

```bash
bash autodl_authority.sh start all
```

脚本保存当前解释器绝对路径，后台进程继续使用该环境。默认输出到 `/root/autodl-tmp/paper4_authority_v1`。执行顺序：验收检查 → 60 次开发学习率试验 → 冻结学习率 → 300 次主实验 → 210 次归因实验 → rule-only → 汇总。60 次开发试验不计入正式 510 次训练。

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

最终汇总位于输出目录的 `summary_main_attribution_rule_only/RESULTS.md`；逐任务检查点、训练历史及逐样本干预记录分别位于 `development/`、`main/`、`attribution/`、`rule_only/`。详细协议见 [PAPER4_AUTHORITY_GUIDE.md](PAPER4_AUTHORITY_GUIDE.md)。

代码同步到 GitHub；实验输出、模型检查点、虚拟环境和患者数据保留在本机/服务器，不通过 Git 同步。

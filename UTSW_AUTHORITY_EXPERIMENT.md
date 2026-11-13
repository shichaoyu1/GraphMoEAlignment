# UTSW 分子证据约束分级实验

## 上传与启动

将 `utsw_authority_server.zip` 上传后解压，进入 `glioma` 文件夹。原始 MRI 和 TSV 留在服务器自己的数据目录，不包含在代码包里。

```bash
unzip utsw_authority_server.zip -d utsw_authority_run
cd utsw_authority_run/glioma
python -m pip install -r requirements-utsw-authority.txt
# 服务器应已安装适配 CUDA 的 PyTorch 2.8.0。
DATA_ROOT=/root/autodl-tmp/dataset/UTSW-Glioma \
METADATA=/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv \
nohup bash run_utsw_authority.sh > launcher.log 2>&1 &
```

修改前两行数据路径即可。默认结果放在 `output/utsw_authority_grade`，进度写在其中的 `console.log`。同一命令可以从最后完成的 epoch 继续；已完成模型自动跳过。代码或规则改变后使用新的 `OUTPUT` 目录，不能混用旧结果。

启动顺序：独立测试 → 全队列影像空间检查和缓存 → 六例训练集小批量学习检查 → 三个低成本对照 → 第 0 折两模型及验证检查 → 其余八次训练 → 汇总、配对区间、出图、打包。任一步失败立即停止。无需手工选择测试集病例或编辑图中数字。

正式训练完成后回传 `output/utsw_authority_grade/results_for_paper.zip`。它包含逐患者预测、实际依赖与状态、完整规则审计、固定划分、汇总及图件，外加所选病例的缓存 ROI；不包含原始体积或大检查点。完整训练检查点留在服务器。

## 固定协议

三分类：Grade 2、3、4。618 例有标签患者参与监督训练及评价；7 例无等级记录只参加数据审计。固定五折，按等级与激活状态联合分层，划分种子 2026。第 k 折测试、第 (k+1) mod 5 折验证、其余三折训练。模型种子 101。匹配模型使用相同初始化、shuffle 生成器和水平/垂直翻转增强。每次训练 50 epochs、batch 8、AdamW lr 0.0003、weight decay 0.0001。验证 Brier 最小的 epoch 为唯一检查点选择依据。

四个独立可训练 ROI2DEncoder 接收 T1、T1ce、T2、FLAIR，每个输入为 7 个 96×96 ROI 切片。复用既有肿瘤包围框、边界扩展和归一化；所有序列及分割必须形状、方向、affine 一致。缓存保存原始文件名、空间坐标、包围框和 z 索引。分割仅定位，并非组织学坏死证据。

与影像空间一致的全零分割使用既有的中心七张完整轴位切片方案，并在影像审计中标为 `central_full_plane_empty_mask`。BT0926 和 BT1090 有参考等级，保留在原定折中。BT1016 无参考等级，仅参与审计。

分子报告为第五个来源。四项三值 one-hot（IDH、1p/19q、MGMT、IDH 方法；第三值表示缺失或未确定），外加适用资格位。适用资格仅由预先指定的三个少突胶质细胞瘤历史类型派生。原始诊断文字、Pathology Information、Grade 不送入模型或编译器。两方法获得完全相同的输入、资格位、规则、投影和损失。

每个来源构造 8×8 均值增广 SPD，转换为 36 维对称 Log-SPD 向量。两层图更新，hidden 64，cross mass 0.35，四个固定于患者的只读支持节点。ACF 激活时报告节点只读取自身和支持节点；MRI 辅助节点和最终读出仍参与预测。learned-joint 放开报告节点读取 MRI 的路径，其他结构保持一致。

损失为维度归一化 Brier（每患者三项均方误差）加 0.1 倍投影前后均方位移。欧氏允许类别单纯形投影对两模型完全相同。

## 临床关系与审计

[WHO 2021 summary, Table 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC8328013/) 将 IDH-mutant、1p/19q-codeleted oligodendroglioma 的等级列为 2、3。[UTSW 数据说明](https://doi.org/10.1038/s41597-026-07274-4) 描述组织分子检测并提供逐例 IDH 检测方式。

激活条件同时满足：IDH 为 mutated；1p/19q 为 co-deleted；IDH 方法为 IHC 或 NGS；历史类型属于 `OLIGODENDROGLIOMA, ANAPLASTIC`、`OLIGODENDROGLIOMA, NOS`、`OLIGODENDROGLIOMA GRADE 2`。这些类型只映射为一个资格位，类型内的等级字样不编码。该保守资格变量来源于诊断记录，应明确披露，不能解释为纯影像场景。激活约束为 p(Grade 4)=0。

本地版本核对：625 例、分级 101/97/420、缺失 7；63 例 IDH-mutant/co-deleted 候选，其中 58 例符合上述范围，5 例保留在预测队列但主分析不激活；58 例的参考等级为 37 例 Grade 2、21 例 Grade 3。等级只用于事后冲突审计，不能决定激活。

接纳的是整理后的检测记录。可用性、IDH 方法、`assay_validity=unreported` 和 `curated_record_accepted` 分开记录；不声称已核查标本质量。明确无效或撤销接纳的记录不激活。撤去 IDH 或方法后编译器撤销关系。MGMT 仅作为预测输入。

## 指标与解释

- Brier、Accuracy、AVR、预测覆盖率、激活覆盖率；Macro-F1、Balanced Accuracy、各类召回及混淆矩阵。
- AVR 是激活且接受预测的患者中 p4>0.000001 的比例，分母同时导出；没有激活关系记 N/A。
- Residual Brier/Accuracy 在激活且参考等级位于 {2,3} 的患者上计算，p2/p3 重新归一化，Brier 再除以 2。若有参考冲突，单列且保留在总体风险中。
- 原始预测与固定 MRI 噪声（标准差 0.1，按患者 ID 固定随机种子，裁剪到 [0,1]）/中央 32×32 遮挡分开导出。指定状态 shift 是同一患者报告向量差的最大绝对分量，再报告患者 mean/max；只在固定激活编译下解释。撤销检测单独记为规则撤销。
- 五折留出患者配对 bootstrap，2000 次、种子 2026，差值方向 ACF 减对照；区间条件于这些已训练模型，不表示多训练种子不确定性。
- 对照为训练集类别先验、先验加投影、C=1 L2 多项逻辑回归分子分类器加投影；仅使用各折训练患者拟合。

这是单队列、单训练种子的回顾性分级实验。MRI 模型与分子记录不一致不等同于两种临床检测结果真实冲突。正确执行已指定规则、保护指定状态和分级预测能力分别报告。

## 正文图与后续论文整合

`aggregate` 要求所有十个正式训练完整且每患者恰有一次测试预测。两行图使用同一患者相同四个 ROI、分子卡片、实际依赖掩码、投影前后概率、预测与单独读取的参考等级，以及两种干预的状态变化。权威来源来自规则；两方法均获得同一规则。标注 M 为报告节点，S1–S4 为只读支持。

从激活、影像完整、ACF 分类正确的留出患者中选 Brier 最接近其中位数者，平局按 ID。不比较基线劣势来挑选病例；如无成功病例则以同样规则选失败病例并在图注明。`figure_provenance.json` 保存每个数字与来源。

生成 `utsw_authority_case.pdf`（矢量文字与图表）、300 dpi PNG、`paper_insert_blue.tex`、`paper_results.md`。短结果段落只陈述实际值，不自动声称优势。`config/utsw_authority_appendix.tex` 是预写的蓝色附录协议。正式结果回传后将正文插入片段与图加入既有蓝色修订副本，并编译核对正文不超过原稿追加一页；原始论文保留。运行服务器实验前不向正文添加性能结论。

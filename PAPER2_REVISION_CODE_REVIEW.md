# Paper2修订实验代码审查记录

日期：2026-10-13。范围：新增 `paper2_revision/`、四个新CLI、独立服务器脚本及新增测试。本次为本地代码审查与回归验证，不冒充外部独立评审或真实医学实验。

## 结论

当前审查范围内未发现尚未处理的阻断问题。旧实验的Python代码、服务器脚本和默认配置无修改；新增入口通过独立输出格式与目录接入。全部115项测试运行结果为通过，其中113项执行成功、2项因缺少可选cvxpylayers依赖按原测试条件跳过。新增17项测试均执行成功。

这证明已覆盖的行为和入口兼容性，不是对所有数据、CUDA设备、多进程环境或科学结论的无条件保证。正式服务器队列尚未跑新补实验。

## 已处理的发现

| 发现 | 处理 | 验证 |
|---|---|---|
| 新目标对象使用tuple，旧anchor_center_loss按list进行张量索引 | 调用既有损失前显式转list，未改旧损失函数 | NIfTI完整训练测试 |
| 改变target policy仍可复用旧direct表征，破坏消融解释 | all_patient_anchors只沿用患者划分，不借region_rules检查点；单次加载也拒绝policy不匹配 | 配置计划与不匹配拒绝测试 |
| 冻结校准可能更新BatchNorm统计 | router阶段除冻结参数外将encoder设为eval | 逐项比较encoder state，包括buffer；prototype完全一致 |
| 若只核对外部anchor_vocab文件，检查点内词表仍可能不同 | 两者都核对，并严格加载模型结构 | reference路径及词表保护；冻结检查点闭环测试 |
| 复用检查点时注册/未注册影像输入偏好可能不同 | 增加prefer_registered一致性检查，拒绝不同预处理的冻结对照 | 不匹配拒绝测试，失败不创建输出目录 |
| 校验失败后可能留下占用输出目录的半成品 | reference加载及校验在创建输出目录前完成 | 失败后输出目录不存在 |
| 路由打乱可能为恒等或包含患者固定点 | 穷举3!−1节点排列；患者随机次序循环移位保证无固定点 | 检查全部映射、重复seed确定性与5种排列 |
| 配对聚合可能隐式忽略缺失运行或不同训练/验证划分 | 要求fold/seed全集一致，核对所有患者membership、正例、候选库，拒绝混合配方及重复患者跨折 | 拒绝目标变更和运行缺失；闭环导出协议检查 |
| 跳过旧结果时可能复用不同版本代码的输出 | skip_completed要求配置、源码哈希及记录的环境版本均一致 | 分支静态审查；输出目录覆盖拒绝测试 |
| 同检查点direct评分可能冒充独立训练对照 | 标明direct_score_scope；只有冻结reference校准可称冻结direct参照 | 输出契约测试；执行说明限定 |
| 外部验证可能污染训练/选择，或混进内部OOF汇总 | 固定原词表和已选检查点，不创建optimizer；检查患者ID交叉；使用external_records独立文件 | 频率及模型冻结转移测试，重叠拒绝及源拟合文件原样验证 |

## 科学协议审查

- 训练频率与静态gate初始化只用训练患者；验证集负责选模型，测试集无选择路径。空验证集直接失败，不以test补位。
- `full_gallery`保留原规则下的全部候选竞争；`observed_fields`是显式不同的候选协议。未知值不当作明确阴性，协议中披露其在全库竞争的位置。
- 正例构造与旧函数一致；空目标、OOV、回退、规则排除的已知标签和空区域均可审计。原26候选不会硬编码到新折或外部队列。
- 多标签字段分类对未知或未选字段不计算交叉熵。Proxy Anchor多标签适配和mean-positive proxy loss明确区别于原论文系统。
- `family_supervision=off`只关闭显式gate家族目标；检索损失及默认within-family损失仍保留。检验更少监督时须显式设lambda_within=0，不能声称无监督。
- family投影、gate和原型评分为无topology因子对照。没有隐藏specialization/balance/distillation。新配方与历史完整MoE有差别，必须使用新矩阵内部配对归因。
- 所有排名指标命名为mAP/MRR/Hit@1；患者先聚合节点，再聚合seed，区间基于患者，不把seed/query计作独立患者。
- 重训练、冻结评分回放和外部推理有独立范围说明。源队列重命名后的重复患者仍需人工核查，代码仅能检查现有ID。

## 实际检查

完整测试命令从包父目录运行：

```text
python -c "import torch, unittest; torch.set_num_threads(1); suite=unittest.defaultTestLoader.discover('glioma/tests', pattern='test_*.py', top_level_dir='.'); result=unittest.TextTestRunner(verbosity=1).run(suite); raise SystemExit(not result.wasSuccessful())"
```

结果：`Ran 115 tests ... OK (skipped=2)`。覆盖既有Paper2、Paper4、TopoMoE、manifold/geodesic、authority、Idea4/5及新增修订模块。两项跳过为authority的可选一般投影依赖，不属于本次新增功能。

独立脚本通过Git Bash的`bash -n`语法检查。新训练CLI帮助可读取；默认矩阵计划实际检查为10次frequency/node_frequency补实验，无训练副作用。`git diff --check`通过；对既有`*.py`和`*.sh`的tracked diff为空。原已存在的Paper4文档改动不属于此次代码修改。

测试NIfTI全部由临时合成夹具产生并清理，没有运行/覆盖真实45次主矩阵。没有将合成测试成绩写入科学主结果。

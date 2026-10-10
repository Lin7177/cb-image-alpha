# I20/R20公开结果

此目录由已验收`course-materials-20261010-r3`聚合素材逐字节导出，历史评价run为`p4-reval-20260919-r3`。`manifest.json`保存文件SHA256、字节数和本机来源相对路径，源文件未随本目录一并公开。事实数值与图表保持原值，没有新训练、推理或原始OOS解码。

[核心源码指纹](source_fingerprints.json)绑定本次发布保留的24项模型/训练/评价/协议/环境文件；发布审计会检查这些字节。它与历史实验验收分别保存，不把发布快照称为新的实验。

- [分类结果](tables/classification.csv)：5成员与集成，含混淆矩阵、accuracy、BA、AUC、CE、Brier及always-up参照。
- [原文汇总](tables/paper_comparison.csv)与[十分位完整11行](tables/paper_decile_comparison.csv)：Table II I20/R20 EW，H-L直接取原文报告值。
- [五成员描述统计](tables/member_dispersion.csv)：均值、样本std(ddof=1)、范围和集成；不是CI。
- [227个月组合序列](tables/monthly_portfolios.csv)：可独立重算集成EW/VW年化收益、波动和SR；累计列是收益差求和。
- [44个epoch](tables/training_history.csv)、[逐年结果](tables/yearly.csv)、[换手](tables/turnover.csv)、[实现差异](tables/experiment_differences.csv)。
- [完整汇总JSON](tables/summary.json)、[原文来源](tables/paper_sources.json)、[展示素材与问答](PRESENTATION_MATERIALS.md)。
- [离线图文报告](report.html)：下载后用浏览器打开；`figures/`含10张PNG。GitHub不直接执行HTML，可在根目录运行`python scripts/paper_demo.py`生成独立离线HTML。
- [分析验收](ANALYSIS_ACCEPTANCE.json)及`evidence/`中的训练、历史重评和组合受限验收。审计状态只表示相应过程完成，不证明严格无泄漏或盈利。

口径：公开分类主结果采用float32五成员概率均值、`p>=0.5`；原论文风格历史列采用float64均值、`p>0.5`，准确率相差1行，不能混用。AUC/Spearman保留已验收历史序号排名ties实现。年化收益为月均×12、波动为样本std×√12；百分比显示需要把小数×100。换手1.742595为174.2595%。原文表数字已经舍入；没有转录原文分类准确率/月频VW，不补造对照。

边界：只覆盖一种配置I20/R20、五成员；`Ret_month`代理收益；真实收益端点未验证；评价已见；未扣交易成本；2019只有11个月；图05为合成示意。没有原始数据、训练权重、行级预测或原论文/第三方代码。

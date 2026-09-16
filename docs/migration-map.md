# 方法与代码迁移地图

审计日期：2026-09-10。仅静态读取参考代码，未执行/导入参考模块。

## 来源固定

- 论文：根目录 `1_The Journal of Finance - 2023 - JIANG - Re‐ Imag in ing Price Trends.pdf`。本文页码以 PDF 第几页计。
- `O` = `Reference_repository/jofi13268-sup-0002-replicationcode/trend_submit/trend_code_submit/`。
- `R` = `Reference_repository/-Re-imaging-the-price-trend/`。
- Reed 本地 Git 来源 `https://github.com/Reed2006/-Re-imaging-the-price-trend.git`，commit `a42b749a3a05efa3c9d769f4f6d77950f55a7bb9`。本地未见 lich99 仓库，本轮不对其做比较结论。
- 两份快照未找到独立 LICENSE；不能推断为 MIT 或允许任意复制。当前选择规范提炼与独立实现，实际复制前补授权记录。

官方关键文件 SHA256：

| 文件 | SHA256 |
|---|---|
| `O/Data/chart_library.py` | `46DE33B6B1D682F13B874BC8D44C59062A4E08C981E42F739655F45E24010036` |
| `O/Model/cnn_model.py` | `33D2D84D2E42F9584AECFF01ECBF72085C6B97F66FDC14C28DC5FD1F85FA51E0` |

## 模块地图

| 模块 | 规范/参考位置 | 本项目动作 | 首期验收 |
|---|---|---|---|
| 数据适配 | `O/Data/equity_data.py`；`R/src/cn_data.py` | 自建只读 CB catalog adapter，不继承CRSP/CSMAR字段与复权假设 | 解码、交易日、单位可追溯 |
| 窗口/标签 | `O/Data/generate_chart.py`；`R/src/cn_data.py` | 自建显式 I20/R5 契约；采样步长和收益端点独立配置 | 时间端点人工与自动核对 |
| 像素渲染 | `O/Data/chart_library.py`、`O/Data/dgp_config.py` | 以官方行为为规范独立实现 | 合成OHLCV像素fixture |
| Dataset | `O/Data/chart_dataset.py`；`R/src/data_utils.py` | 借鉴memmap思想，自己维护样本索引与分片 | 图像与sample_id一一对应 |
| CNN | `O/Model/cnn_model.py`、`O/Misc/config.py` | 提取20日配置，逐层形状核验 | 输出两类logits、保存加载一致 |
| 训练 | `O/Experiments/cnn_experiment.py`；`R/src/training.py` | 自建简洁训练器与最佳权重恢复 | 早停只用验证集、checkpoint对应正确epoch |
| 预测与指标 | `O/Misc/utilities.py`；`R/src/training.py` | 自建可追溯预测表与基础分类评估 | 有限概率、基线与类别分布 |
| 组合/回归 | `O/Portfolio/`、`O/Analysis/`、`R/src/backtest.py` | 暂缓 | 不纳入首期依赖 |

## 论文原设定和本项目选择

| 项目 | 论文/官方 | 本项目 I20/R5 |
|---|---|---|
| 图像 | PDF pp.6–9：每天3像素，黑底白线、MA与成交量 | 保留；I20 为64×60 |
| MA | PDF p.8：MA窗口与图像长度相同 | MA20，额外历史warm-up |
| 收益 | PDF p.14：基于CRSP调整收益重建相对价格 | 不继承；待核验本地复权，以明确价格收益口径起步 |
| 标签 | PDF pp.10、13–14：正收益1，否则0；I与R可独立 | I20/R5，零收益为0，缺标签排除 |
| 模型 | PDF p.11：I20三层CNN block | 官方20日结构；具体padding/尺寸以源码核验 |
| 训练 | PDF p.13：Adam lr=1e-5、batch128、Xavier、BN、dropout0.5、验证loss连续2epoch无改善早停 | 作为基线参数；若显存调整batch必须显式记录，不能冒称完全同设定 |
| 像素标准化 | PDF p.13脚注8：训练集均值和标准差 | 保留，只在purge后的训练图像拟合 |
| 切分 | PDF pp.12–14：1993–2000内随机70/30，2001–2019独立测试 | 改为转债日期顺序train/val/test并purge跨界标签 |
| 缺失 | PDF pp.7–8：允许中间空白/部分空白列 | 首版完整窗口排除策略；记录偏差与数量 |
| 重复训练 | PDF p.14：每配置5次平均 | 首次单种子，后续再扩展 |
| 结果分析 | PDF p.14起：十分位及多空组合 | 首期仅预测和分类指标，回测后置 |

步长、日期边界、定点解码、价格调整仍须在实现前冻结。不能因选择I20/R5就声称论文完整复现。

## 为什么不直接采用 Reed 骨架

1. `R/scripts/cn_train.py:52,55` 对重叠窗口随机拆分train/test和train/val。它不能充当本项目时间外验证协议。官方仅在历史训练期内随机切分，两者不能混同。
2. `O/Data/chart_library.py:52` 的I20价格/间隔/成交量分区为51/1/12行，`R/src/cn_data.py:13–14` 使用49行价格区。官方 `:149` 是单像素量柱，Reed `:144` 是三列；官方 `:171` 将MA跨日连线，Reed `:137` 仅散点。
3. `R/src/cn_data.py:116` 附近零量日跳过价格绘制，需要重新定义停牌处理。
4. `R/src/models.py:22` 的padding随dilation放大；官方 `O/Model/cnn_model.py:62` 不按dilation放大。初始化bias也有差异，不能当成完全同一模型。
5. `R/src/training.py:82` 用 `best_state = model.state_dict()` 保留内存引用，后续训练会改变其中tensor；落盘checkpoint与末尾内存恢复要区分。官方 `O/Experiments/cnn_experiment.py:508,571` 使用deepcopy。
6. `R/src/cn_data.py:184` 标签字段固定名 `Ret_20d`，`R/src/data_utils.py:15–18` 固定64×60；直接更改窗口参数不保证正确。

## 环境与副作用

官方 `cnn_env.yml` 是Python3.8.13/PyTorch1.7.1等历史环境，含回归、notebook等首期不需要的依赖；不能直接拿来适配本机新GPU。Reed仅README列依赖，没有锁文件。

`O/Misc/config.py`、`O/Data/dgp_config.py` 在import时创建工作/缓存目录，生产包不得直接import。官方README说明缓存样本只保留首末各50行，缓存ZIP不能当完整训练集。

## G1迁移政策补充（2026-09-11）

转债数据采用已验证catalog价格与存储数量，EOD为显式研究假设，不复刻CRSP总收益或声称历史真实到达。原始物理单位及上游不可逆量化限制保留；I20/MA20/R5的预备窗口政策见[架构数据政策v1.1](architecture.md)，变更依据及双独立交叉审计记录见[主控](plans/2026-09-10-foundation.md)。这属于本项目调整，G2尚未实现。
G2 v1.2已具体化本项目时间顺序迁移：2020–2023训练、2024验证、2025至2026-07测试；用户已批准，属于本项目迁移设定，非论文原始设定，详见architecture末节及主控交接。

## G3 I20实现审定

静态规范源仍为固定官方chart_library.py/cnn_model.py及Misc/config.py；Experiments/cnn_experiment.py:1042–1055确认元组次序为filter/stride/dilation/pool。未导入或复制参考源码，许可未明限制保持。
本项目render_version=i20-v1，model_version=paper-i20-v1，像素及逐层规格集中在architecture的G3节。保留51/1/12、单列量柱与最低量柱裁剪、MA连线、ceil池化及非dilation扩张padding。
主动差异：沿用已批准CB原始catalog价格/MA而非CRSP总收益重建，G2完整窗口/每日候选政策不变；全退化价格显式失败不静默删除；不执行官方初始化随机dummy前向，因此BN初始统计保持mean0/var1/count0，不复刻其随机数消耗。两位G3审计认为后者为独立实现差异而非主线/数据协议实质变更，已补初始BN测试。
数值限制：float64/rint与官方舍入惯例一致，但MA计算顺序/上游价格不同，不宣称任意窗口全链路位级一致。合成二进制可精确表示的比例fixture验证半像素ties-to-even，真实窗口另核数值坐标。
CNN手算记录：首层有效核高9，floor((64+4-9)/3)+1=20，pool10；后层conv10/pool5、conv5/ceil-pool3。Flatten46080，总参数708866。首轮fixture曾误写pool11/6/3，按该独立公式纠正；并非更改网络去满足错误测试。
## G4 smoke范围

沿用论文候选Adam lr1e-5、batch128、验证loss patience2和既有CNN；本次seed20260912、Train256/Validation128按sample_id升序选择、最多4epoch只为闭环。归一化仅拟合这256张训练图，不能复用为未来G5全量训练统计。未进行Test评价或超参搜索。第一轮因工程验收补强被superseded，同配置重跑非模型选择；正式smoke入口见主控。
G5已获批准，采用固定全量G2范围和最多20epoch单seed正式实验，不继承smoke子集、权重或归一化；相对论文差异及资源定义见architecture末节和主控G5提案审计记录。

## 后续研究路线草案（research-v2，待执行）

G7开发区间产物诊断、G8固定三折数值/量化logistic与CNN比较、G9条件性单轴判别属于新研究协议；fold/数值特征/epoch内层选择均是本项目研究调整，不是原论文设定。G0–G6历史产物冻结，Test已查看，2024已用于选择，后续只形成开发证据。尺度分支若批准另记feature-v2.1，标签/交易假设不默认改变。路线、预算、双审处置和执行授权以主控及architecture为准。

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


## R2原论文复现差异登记（paper-v1 DRAFT，2026-09-16）

本节使用本机已提供论文PDF页码，官方快照只静态读取。主论文事实优先于示例脚本默认值，尚未冻结的选择不称论文事实。新分支不修改CB原配置/模型/数据适配器；没有参考源码复制许可，独立实现。

| 维度 | 论文事实 | 官方代码观测 / CB差异 | paper-v1拟定处理 |
|---|---|---|---|
| 市场/年份 | PDF p14：CRSP、NYSE/AMEX/NASDAQ，1993–2019；OHLC自1992年6月起 | CB为2020–2026可转债，不能代表原市场 | 作者公开monthly_20d包目录覆盖1993–2019；原CRSP权限和九组数据可得性待核 |
| 图像/缺失 | pp7–8：RET重建相对价格，OHLC按当日close比例；MA同窗；缺失中间列保留，IPO/退市穿窗排除 | CB直接目录价格且完整窗口筛选 | 优先使用作者已生成图像；这能复现输入，不等于已验证原OHLC到图像的全链路。原始数据分支待数据 |
| 年份切分 | pp12–14：1993–2000池内随机70/30，2001–2019固定OOS | CB顺序split；代码random_split在ConcatDataset上 | 保留论文池内随机，不跨训练池/OOS边界随机；内部重叠泄漏风险明示，不以CB顺序规则替换论文实验 |
| I/R范围/集成 | p14：I5/20/60×R5/20/60，五次独立训练概率平均 | CB单I20/R5、单seed | 最终目标九组×五成员；公开入口仅I20月频，不能据此声称九组完整 |
| 预测/标签频率 | p14：预测与持有为周/月/季（约5/20/60日） | EquityDataset.ret_val_name用Ret_week/month/quarter，而不是Ret_5d/20d/60d；GenerateStockData基于期间结束日 | 首个可得配置I20/R20使用月末图像+Ret_month，明确区别固定20交易日。不能用月末图像+Ret_5d冒充周频I20/R5 |
| 图像归一化 | p13脚注8：训练图像所有像素mean/std，验证/测试复用 | chart_dataset.py:180起按每年头50000幅（过滤前）计算，各年包括OOS；与论文矛盾 | 遵循论文，仅随机70%训练子集统计并冻结；不采用OOS分布，记录与官方实现差异 |
| 训练参数 | p13：lr1e-5、Adam、batch128、Xavier、BN、dropout0.5、patience2 | sample_codes.py用1e-4，Experiment默认1e-5；代码max_epoch50，CB20 | 按论文lr1e-5；最大50来自实现而非论文，若预算未完成视未完成，不缩样本凑PASS |
| 训练边界 | 论文仅明确年份；代码remove_tail在2000用12/24、12/1、10/1严格小于 | CB按label_end精确purge | 对公开I20/R20拟保留2000-12-01前样本，但这仅是官方边界代理，不能证明收益端点未进入2001；端点无法核验时阻断训练并请求协议裁决 |
| 评价 | p14起十分位EW/VW、H-L、SR等为主要实证 | CB主要分类指标 | 新评价协议逐表记录，不能仅以分类AUC代替论文结论；权重/收益/频率/退市处理未明确则不运行 |

来源：作者主页 https://dachxiu.chicagobooth.edu/ 与Image Demo https://dachxiu.chicagobooth.edu/download/img_demo.html 提供图像/字段入口；公开ZIP https://dachxiu.chicagobooth.edu/download/img_data.zip。2026-09-16 HEAD和64KiB尾部范围请求核验112个ZIP目录条目、约8.54GB未压缩，包含1993–2019的monthly_20d；元数据尾部检查不等于内容/CRC验证。各年图像/标签尚待逐项审计；不再分发作者数据。官方README明示随代码缓存仅保留首末50行，不能作为完整训练集。

### P1独立审计修订（2026-09-17，仍IN_REVIEW）

- 官方`chart_dataset.py:164–176`分类路径先执行`ret > 0`再处理NaN，缺失收益已变为负类。paper-v1必须先检查有限Ret_month、排除-99占位；零收益归0，不按收益大小/未来波动筛选。缺失、占位、EWMA非有限/非正、2000尾部应分项且按固定优先级对账；现阶段审计候选计数不是最终样本名单。
- `equity_data.py:115–124`的Ret_month来自每标的下一条可用期间末记录，缺月可跨多月；公开记录的图像Date不是收益结束日。仅凭文件年份和2000尾部代理不能证明label不跨OOS。未证实实际污染，也不能当成无污染；缺端点时训练入口BLOCKED，不能通过读2001–2019标签来补证。
- 每成员独立随机70/30拆分符合官方每次重建loader的行为，具体RNG未由论文给定。拟定先固定过滤与样本顺序，以floor(0.7N)训练，每成员独立split/norm、模型/shuffle随机流，保存各名单和摘要；各成员只以自身验证loss恢复最佳，全五等权平均不择优。确切seed与实现契约须在P2前审定，未运行。
- P1冻结仅覆盖已核验的输入/开发契约；P3资源、次数、恢复政策及P4具体表格/权重/缺失/并列分位/年化在各自启动前另行冻结审计。50epoch是代码上限，达到上限而未早停须标capped。九组五成员是核心预测范围，不等于全部国际样本、稳健性和交易成本表已复现。
- 完整ZIP允许下载和摘要固定，但只按allowlist解压1993–2000，逐成员CRC检查；不全包testzip。开发池包含未来30%验证，因此不称纯训练子集。下载后的SHA不能证明该文件与论文发表时相同；作者公开下载入口不赋予再分发许可。


2026-09-17实证更新：条件续传及开发池审计完成，八年实际schema无收益端点字段/原始期间序列；端点verdict为不可可靠验证，P1继续BLOCKED。已核验16个开发文件CRC/SHA，不再处于“仅ZIP目录元数据”状态；未对OOS内容做同类核验。处理方案与忠实度/右边界保证差异集中见data-audit，未批准任何新样本/标签协议。


B用户裁决后paper-b-v1已冻结：Date代理风险被接受，非严格leakage-free；保留finite标签修正、train-only逐成员归一化、现代自有CNN/RNG实现偏离。P2已形成实际778507样本及5组固定70/30索引，未训练；不会把本分支描述为逐bug、原始OHLC全链路或九组全论文完整复现。验收/下一门以主控最新交接为准。


P3执行层选择已单独冻结paper-p3-training-v1：确定性CUDA/TF32关闭/Adam foreach与fused关闭、独立验证loader RNG、硬停止预算、完整epoch恢复及capped状态均是本项目现代可复现执行设定，非论文精确环境或随机轨迹事实。它们不改变B样本/标签/70-30/五seed，但仍不能声称原论文逐位复现或严格无泄漏。当前未训练，GPU恢复仅为待实施验收要求。

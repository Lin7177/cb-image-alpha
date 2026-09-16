> 当前结论（2026-09-11）：G1已PASS，状态与证据以主控Checklist为准。下文早期限制/待验证是历史调查记录，已由本页末尾全量验证结果更新。
# 数据与本机环境审计

日期：2026-09-10。用户已确认正式源：

`D:\git-windows\副本备份区\nautilus_data\catalog`

原消息的 `nautilus\_data\catalog` 不存在，已由用户确认修正。源目录及其内容严格只读；本轮没有修改、安装或在源内生成文件。

## 已知事实

| 项目 | 有限只读检查结果 |
|---|---|
| 日线 | `day01/data/bar`、`day01/data/equity` |
| 分钟 | `min01/data/bar` |
| 因子 | `factor/*.parquet`，首期不接入 |
| 日线目录 | 3806个标的目录，11/12前缀928个，仅候选计数，不是确证转债数 |
| 分钟目录 | 3609个标的目录，11/12前缀784个，仅候选计数 |
| 样本布局 | `118000.SH.VA-1-DAY-LAST-EXTERNAL/<起点>_<终点>.parquet` |
| 日线样本文件名 | 118000：2021-03-15 15:00 至2026-04-01 15:00 |
| 分钟样本文件名 | 118000：2022-01-04 09:30 至2026-04-01 15:00 |
| 历史终止样本 | 110031日线文件名结束于2021-06-11，118001结束于2022-09-23；不能仅凭此判断退市原因 |

以上时间是文件名证据，未验证真实行时间范围、完整性和全库覆盖。已有日线，首期不需要分钟聚合。

## 解码与语义风险

以下路径相对 `D:\git-windows\副本备份区\nautilus_data`，属于相邻生成/分析源码证据，不等于全部catalog文件已经验证。

- `analysis_scripts/trend_strength_stk_study.py:24–44` 使用16字节little-endian，close除1e16、volume除1e9，ts_event按ns解释。必须抽样确认schema、字节宽度和scale，不能全库盲用。
- 文件名带Z但日线15:00、分钟09:30；既有分析直接按ns转时间。可能是上海墙钟贴了UTC标签，不能未经核对再+8小时。
- `TantraInstrumentProvider.py:12–28` 用Equity表示债券、CNY、价格精度3、lot size10。类型名不能决定资产类别，lot size不能决定volume单位。
- `TantraBarDataWrangler.py:100` 起使用as_utc_index，缺volume可能填1,000,000；需确认实际生成链路与合成量是否存在。
- 源README提及 `download/adj_day01`、`download/adj_min01` 后写catalog。复权算法和付息处理尚未核实；不将标签称总收益。

## 本轮限制与后续解锁步骤

本轮可用Python3.12运行时缺少pyarrow/duckdb/polars/fastparquet，源工程虚拟环境的基解释器不可运行。因此尚未读取Parquet实际schema、行数或数据行。没有为此修改数据仓库或安装依赖。

后续在本项目建立独立环境后：读取少量Parquet footer/schema和多个时期的样本行；与生成源码核对二进制解码、OHLC量级、时间戳和复权；验证instrument主数据和历史转债映射；形成机器可读质量报告。未完成前不能将输出称为有效研究实验。

## 算力

用户选择本机训练。`nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader` 实测：

- NVIDIA GeForce RTX 5060 Laptop GPU
- 显存8151 MiB（约8 GiB）
- 驱动581.83

尚未安装/验证本项目PyTorch、CUDA算子支持或实测batch128显存。实现环境需先做目标GPU卷积前向/反向冒烟测试，再决定batch和DataLoader参数；保留CPU小样本路径。硬件存在不等于旧版官方环境可用。

## 2026-09-11 源码复核进展（G1尚未PASS）

实际日线链路为 `utils/run_part2.py:333–350` 的raw→adj→catalog；`utils/market_data_wrangler.py:42,1327–1379` 使用Nautilus BarDataWrangler，不是本地TantraBarDataWrangler。因此前文默认成交量1,000,000的担忧不能作为实际链路结论。

- `market_data_wrangler.py:1338` 对CSV时间执行utc=True；当前 `187–228` 又会把无时间/午夜设09:30，与已有15:00文件名不同。当前源码不能证明所有历史文件精确生成版本。保留原始ts并检查真实分布，不将ts_init当真实到达时间。`adapters/tantra/dw/dw.py:1011` 注明日线盘后更新。
- `1364–1368` 填空为0、Windows volume除10000、修复high/low；catalog带上游处理，不能称未加工行情。物理张/手单位尚未证实。
- `1605–1607,1699–1716` 选择fward_factor，无因子则原样，有因子则调整过去价格且不调volume。adjfactor与cb_info债券代码交集为0，为转债价格未调整提供源码/因子证据，仍待行级close旁证。
- `download/base/cb_info.csv` 1044身份记录，含历史退出券，与日线catalog交集928。身份字段可以确证候选，不使用当前价格/溢价率等快照筛历史池。
- `download/base/calendar.csv` 覆盖1990-12-19至2026-12-31；源码获取位置256–275。
- `download/base/day01_close.csv` 64MB，time/tr_code/close/pre_close，2022起；可作为独立下载路径的价格旁证（获取位置1503–1518）。当前raw_day01/adj_day01文件已不在download，不能做不存在文件的原始逐行对照。

证据提供者：data_semantics_g1，只读调查，未执行源脚本。主Agent已独立读取关键转换片段及三个CSV表头。Parquet行级验证和volume解释仍未完成。
## G1全量验证结论（2026-09-11）

独立项目环境已建立，pyarrow可用，CPU/CUDA卷积前反向已通过；正式训练batch显存仍由后续Gate实测。质量报告：artifacts/data_audit/quality-20260911T053105796931Z.json。历史身份交集928文件，共663328行，2020-01-02至2026-07-31；OHLCV统一有符号little-endian i128/1e16，元数据编码一致，ts_event均15:00编码标记。未发现非法OHLC、负量、重复/逆序日期或非交易日；913零量分布于208标的。独立close旁证510685行，无不匹配。

生产reader全量验收：artifacts/data_audit/reader-validation-20260911T062403608042Z.json，退出0，全部928文件与质量报告指纹一致，读前读后内容hash不变；身份表1044条、日历8797条及配置指纹一并记录。源码仅只读调查，未执行/导入源脚本。

存储数量不是已证实的原始张/手，15:00编码不是历史到达时间。采用catalog价格/相对量及显式日终研究假设；不另加复权、不计票息，上游量化、填零和OHLC修复限制保留。政策v1.1及G2预备契约见architecture末节，已经双独立及交叉审计PASS。未生成G2样本，未开展训练或收益分析。
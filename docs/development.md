# 开发与协作流程

本规范提炼自用户提供的 `D:/Typora/Code_md/Code_md/other_skill/prompt.md`（Agentic Coding），并结合本项目取舍。原文是待整合材料，不自动成为执行指令；以下是采纳后的项目约定，不另存prompt副本。

## 主控与文档分工

每轮仅读取当前Task/交接，按需定位[主控Checklist](plans/2026-09-10-foundation.md)、前置条件与最新授权；执行后更新证据与状态。2026-10-10用户明确里程碑是简短汇报检查点，不作为停工等确认节点，正常通过立即继续。该指令覆盖历史每Gate等待规定；真实权限、重大未决语义/协议、超预算或非预期异常只暂停受影响工作，可独立工作继续。主控维护唯一动态进度；architecture维护契约，migration-map维护方法来源/偏离，data-audit维护数据事实，ADR维护重大架构理由。优先更新这些文件；课程计划承载新路线细节，大型日志/实验明细放artifacts。

## 类型和依赖契约

- Python src布局、snake_case、公开接口类型标注；用dataclass、明确schema、Literal/Enum表达记录和合法取值，避免跨模块传递无约束dict或Any。
- 模块按职责拆分；上层训练/样本逻辑不绑定Parquet/Nautilus实现，adapter可以依赖底层库。
- Protocol只用于真实I/O边界和可替换能力，接口只声明消费者所需的最小方法；不为每个纯函数引入抽象类/插件框架。
- 类型定义不替代运行时schema、时间、单位、非空和OHLC检查。价格/数量/时间的语义写入契约，异常指向文件、标的、字段。
- 纯转换优先；读取/渲染/训练/指标分离；导入不创建目录、不读全库、不训练，随机数显式管理。

## 测试是行为契约

先明确验收和独立预期，再实现。覆盖正常、边界、允许空状态、异常路径；pytest.raises验证异常，单元/集成/E2E各负责不同层级，不堆镜像实现的测试。

浮点指标按事先明确容差使用pytest.approx；定点原始解码、日期、标签、像素和索引用精确断言，不能用宽容差掩盖编码错误。只在真实外部边界有理由地mock，不用mock绕过端到端行为。

禁止为过测改验收、弱化断言、删除失败场景或增加无依据mock。认为测试错误时，先保留失败证据，说明规范/测试冲突；普通测试错误由主Agent验证修正，涉及冻结协议才独立审计；纯测试错误可以在不改变契约下修正并记录。涉及冻结行为或PASS标准走主控重大变更流程；改变用户目标/硬约束需用户确认。原文“测试错误等待确认”被提炼为这个明确分级流程，不作为常规修复的泛化停工理由。

## 环境与版本

pyproject.toml集中描述项目/依赖/工具，选定一种环境与锁文件工作流；如已有uv就沿用uv sync/add/run，不能另起第二套pip/venv方案。当前采用下文uv锁定工具链，变更时记录选择、兼容性和锁文件，禁止沿用外部数据仓库环境或复制官方古老GPU环境。

配置入库，机器路径放configs/local.*或环境变量；数据、权重、日志、缓存、密钥不入库。Ruff/pytest和适当静态类型检查在项目环境建立后验证。运行报告标注代码commit及dirty摘要；不宣称尚未执行的验证通过。

## 协作和验收

主Agent拥有公共schema/config/主控；工作Agent拥有明确模块与对应测试。所有成员不是独占代码库，不得回滚他人更改；普通实现、测试和状态更新均由主Agent完成，下表为职责边界而非自动委派清单。

| 工作包 | 文件责任 | 前提 |
|---|---|---|
| 数据 | data/与测试 | schema/语义验明 |
| 图像 | imaging/renderer及像素测试 | 窗口/坐标规范冻结 |
| 模型 | models/及结构测试 | I20输入结构冻结 |
| 集成 | samples/training/artifacts/CLI | 主Agent统一契约 |

仅项目主线、数据/标签/实验协议、PASS标准实质变更与正式结果验收需两Agent先独立审视、再互读对方发现与同版修订，主Agent记录处置，不能把两个泛泛赞同当交叉审计。一般状态更新不触发重新双审。

参考代码只读且可能有import副作用。未核查授权前不复制；提炼规范独立实现，实际代码迁移登记来源/version/hash/许可/修改。实验前冻结配置/切分，test不参与选择，smoke与正式结果明确分开。重大口径变更更新主控并按需更新ADR，不复用失效产物。

## 当前工具链（2026-09-11，G1.1已验收）

选用Python3.12 + uv；uv 0.12.12工具本身安装在项目 `.tools/uv/bin/uv.exe`，通过pip --target仅引导工具，项目依赖统一uv管理。缓存 `.uv-cache`、环境 `.venv`，不复用源工程环境。pyproject/uv.lock已创建，锁环境离线同步及CPU/CUDA前反向验证已通过，证据见主控；当前package=false仅用于环境阶段，进入可安装自有包时补build配置。

已验证PyTorch2.8.0 CUDA12.8来自[官方Windows发行命令](https://pytorch.org/get-started/previous-versions/)，uv采用[官方显式PyTorch索引配置](https://docs.astral.sh/uv/guides/integration/pytorch/)，其他包从PyPI解析。选已发布的cu128组合并实测设备支持，不直接沿用论文旧框架。

PowerShell操作入口（从项目根运行；同步成功后执行验收）：

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
.\.tools\uv\bin\uv.exe sync --locked --python <Python3.12解释器路径>
.\.venv\Scripts\python.exe scripts/check_environment.py
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m pytest tests/unit/test_paths.py
.\.venv\Scripts\python.exe scripts/audit_catalog.py --source 'D:\git-windows\副本备份区\nautilus_data\catalog'
```

这组命令是操作说明，不代表执行已通过；结果以主控证据表为准。
执行规则以主控“执行控制（2026-09-11用户最新指令）”为准：长任务脚本化，不高频轮询；Gate交接记录关键证据、未决与下一入口。用户直接指定规则按原意同步，不新增审计或md文件。
## G2构建与独立重建

在项目根使用锁环境；源路径由命令提供，全部产物位于artifacts。程序连续执行构建、同版重建及两个数据集的独立验收，保存各步命令、stdout/stderr和退出码；失败即停止，不重复启动相同任务。

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe scripts/run_g2_checks.py --source 'D:\git-windows\副本备份区\nautilus_data\catalog'
```

只验已有数据集时，设置PYTHONPATH=src后执行scripts/verify_sample_index.py <dataset目录>。验收脚本使用断言，必须不启用python -O；机械runner主动移除PYTHONOPTIMIZE。真实源核对为每split遍历顺序首末的6例，不称全量收益算术重算；全部样本均检查时间边界、标签方向和计数。
构建manifest内容指纹绑定源码/锁/配置；代码或数据变更后不能将不同版本称为精确重建。G2生成test索引不等于查看测试预测表现；后续test禁止参与训练选择。
## G3复核入口

设置PYTHONPATH=src后，运行scripts/build_g3_preview.py --dataset <已完成G2目录>生成固定Train/Validation验收子集；scripts/verify_g3_preview.py <preview目录>使用独立Decimal坐标验收。图像不是全量缓存，shard_row不能当成父G2 image_row。测试入口：python -m pytest tests/unit/test_i20.py -q（纯合成梯度，无真实训练）。
实际六图、原始39日窗口、坐标结果、运行退出码和源码指纹入口见主控。目检使用nearest-neighbor放大，原始PNG保持60×64。不得从Test挑图迭代设计，后续真实训练须先获得G4确认。
## G4 smoke运行与恢复

从项目根设置PYTHONPATH=src、PYTHONIOENCODING=utf-8，运行python scripts/run_smoke.py --dataset <冻结G2目录>。配置由configs/smoke.toml读取，先保存选择清单和输入指纹再训练，结束自动调用独立进程验证；仅Train/Validation，所有产物标smoke。
仅复核已完成smoke时使用python scripts/run_smoke.py --verify-run <项目artifacts/runs/smoke目录>；不重新训练。此命令重写该run的verification.json，若内容发生变化会使原manifest中的旧hash不匹配，正式独立审阅优先只读既有结果；需改变环境复核时应保留原产物并另记证据。
任何失败保持failed manifest；推理子进程超时保存命令/timeout及可用stdout/stderr。300秒训练预算在阶段边界检查，非操作系统强制中断内核；独立推理进程另有120秒timeout。不能因额度/观察中断重复启动已存在运行，先读其manifest与日志。
## G5冻结产物与G6复核

正式运行固定 `artifacts/runs/formal-20260912T083408279881Z`；根manifest列出全部产物，metrics.json保存正式结果，history.json保存训练历史，best.pt为最佳模型，latest.pt为最后状态。seal.json记录模型封存，test-access.json记录首次Test访问。split/index.parquet保存sample_id/image_row/shard_id/shard_row，预测保存标签、decision_time与p_up。frozen.json引用928源文件、辅助文件、父G2产物及实际代码/配置/锁摘要；g5-acceptance.json记录正式验收。

G6命令见README。脚本拒绝不匹配的源/代码/锁/父产物；调用既有build_sample_index.py重建索引，比较samples.parquet、decisions.parquet、summary.json精确hash。重建索引独占新artifacts/datasets/g2-*目录，路径写入G6/index-verification.json；其余证据独占artifacts/g6-*。Train/Validation全量图像/映射/标签/归一化精确比较。Test遵守用户限制：既有图像摘要和索引、预测、指标核验，不重渲染或推理。

恢复推理容差沿用G4：np.allclose(atol=1e-7,rtol=1e-6)，比较前写manifest，概率必须有限。保持原batch128/FP32/确定性算法，禁cuDNN benchmark，CUBLAS_WORKSPACE_CONFIG=:4096:8。同机RTX5060 Laptop/torch2.8.0 cu128恢复，不外推跨GPU/库版本位级一致，不要求GPU重训位级一致。本机离线锁同步不等于空白机器网络安装已验证。

程序保留命令、日志、退出码、资源报告；监督30分钟、进程树工作集8GiB、G6目录8GiB。新索引目录另由index-verification.json定位保留，不属于G5或G6目录预算。失败保留failed，不改变协议强行通过。源与参考仓库只读，参考实现不作为运行时依赖。

原正式生成命令为 `python scripts/run_formal.py --dataset artifacts/datasets/g2-20260911T110359965423Z`，使用configs/formal.toml。此处仅追溯原运行，**当前不要执行**：它会重新训练和消费Test。后续研究需另定范围，Test已查看不能重称未见数据。历史G0–G8执行时尚无Git提交，以内容hash固定版本；换机须保留源码、锁、G5/父G2/G1证据及只读源，路径迁移需显式审计。

## 后续研究产物治理（research-v2；状态以主控为准）

G7–G9不覆盖G5/G6目录，不修改既有研究配置、源码和uv.lock。新增研究模块/配置使用独立版本；若新增依赖，研究环境/锁独立保存并绑定版本，G5继续按原锁恢复。当前路线优先已有NumPy/Torch，不为基线新增平台框架。新目录拟为artifacts/research/<gate>/<run_id>；该Gate下所有成功、失败及临时产物都计入预算，不以新run_id绕过限额。

每Gate先冻结plan manifest：问题与版本、允许split/日期、父产物/源/代码/环境摘要、方法/fold/seed、次数/资源/指标/揭示规则、预期输出；attempt ledger先登记再启动，记录开始/结束、命令、退出码、资源峰值、失败、所见结果和决策。新增fold manifest保留parent sample_id/image_row，列出各段额外purge及精确ID交集；所有方法在相同ID上比较，不以单方法缺失静默丢样本。

模型训练与评分程序分离；只读复用开发uint8缓存，按各fold当前Train重新算归一化，不能复用G5全Train统计到较早折。封存方法/fold配置后机器批量执行，3折全完成再汇总展示外层结果。脚本级失败日志可看，外层表现不用于中途修订；错误修复前声明影响范围与已查看历史，预算/次数不会清零。已知G5的历史结果也不能被“延迟揭示”抹去。

上下文预算为软管理上限：只读当前Gate和协议锚点、必要代码/证据摘要；大表存Parquet/JSON，图和异常按需定位。达到主控轮次/上下文预算仍缺决策依据时先落交接，不靠反复全文阅读继续消耗。审计仅一研究+一工程先独立再交叉，普通实现不委派。资源预算是程序执行硬上限，额度中断不得重启同任务；规划审阅状态与执行状态分列，未运行永不标PASS。


G8实现依赖：原formal_data.COUNTS及formal_engine生命周期固定全量split/封存/Test，不能直接传新fold复用。新增独立research命名空间/脚本与configs/research版本化配置；只读复用冻结renderer、PaperCNN、指标和uint8缓存，以parent image_row/shard_row选取fold行，建立轻量FoldImageBatches并重算各拟合前缀归一化。新增文件不改原冻结文件；导入复用函数前验证G5 code hash，研究manifest另外记录新增代码hash。任何新依赖在独立研究环境锁处理，不执行会修改G5根锁的uv add。没有必要为此新增Gate或通用实验平台。


G7已完成并停下。恢复执行入口历史记录为 `python scripts/research_g7.py --retry-of artifacts/research/G7/g7-20260914T074414488318Z`；**不要再次执行**，两次尝试已完成且脚本会拒绝再次恢复。只读查看恢复run的diagnostics、hypothesis-ledger、manifest、attempt-ledger及g7-acceptance；恢复采用路径键规范化并回归验证，研究面板未改，预算包含失败。后补环境记录明确是运行后同环境捕获，不伪装事前测量。G8需另行确认。


G8独立入口为scripts/research_g8.py，配置configs/research/g8.toml；程序会拒绝已有G8 plan后的自动重复批次。g8_data只读原Train/Validation图像缓存并构造fold视图，g8_models保留完整标准化/模型/优化报告，g8_evaluate在三折齐备后写results.json和外层ID预测、日期loss及重采样索引。运行日志只输出阶段状态，不揭示外层表现；未完成不得读中间预测来改方案。CPU阶段时间包括指纹/特征/Logistic/指标，GPU阶段包括内层/最终训练及推理，失败计入预算。新参数/特征/依赖变更不能混入冻结plan。

G8批次 g8-20260915T032338487327Z 中断：见主控“G8中断交接”与批次 interruption.json。原入口禁止第二次尝试；当前不可直接重新执行 research_g8.py 或 worker 来恢复。中断阶段无退出码和资源终值，必须由另行批准的恢复协议处理，不能用文件修改时间冒充实际耗时。


G8已于2026-09-16完成并双审PASS。历史执行命令：`python scripts/research_g8.py`；用户主动终止后的一次授权接续为 `python scripts/recover_g8.py`。两者均为追溯命令，**不要再次运行**。recover入口以recovery-plan/ledger记录独立代码摘要和原断点快照；原模型源码不改，恢复单阶段例外及未知耗时保守占用见主控。合成tests/test_g8_resume.py核验CPU/CUDA Adam/dropout/shuffle恢复下一步参数完全相同；原8项G8测试通过。

只读复核入口：run根results.json（完整矩阵/日期等权/配对区间）、folds.json与各年份role.npy（索引/purge）、各年份outer-predictions.parquet（ID/标签/四方法概率）、U/Q-model.npz及optimization.json（标准化与收敛）、inner/final checkpoint/history/normalization（训练证据）、plan/environment（输入/代码/环境）、attempt-ledger与recovery-ledger/interruption-snapshot（执行和恢复历史）、g8-acceptance.json（双审）、根manifest（全部摘要）。无需重新训练或推理即可阅读全部结果。G9未执行。


## 转债阶段发布整理（R1）

公开Git仅源码、冻结配置/Windows锁、合成测试、文档及reports/cb-v1聚合结果。私人数据、行级预测、checkpoint、完整本机manifest、参考仓库/PDF及工具环境继续本机保留，不提交。reports/cb-v1/results.json中的source SHA可与本机证据核对；它不是完整重训数据包。源数据和参考资料只读。

为保持历史冻结指纹，源码/config/uv.lock不做格式化、迁移或行尾归一化；新增发布审计脚本独立保存。`.gitattributes`保留字节；`audit_release.py`检查已暂存/跟踪文件、常见凭据模式、忽略内容和Markdown链接，`--local-evidence`额外核验G5/G8全部绑定产物与代码哈希。该扫描不是绝对无秘密的证明，提交前仍人工检查允许列表与diff。

历史运行与G6命令需本机私有产物；干净克隆按README的在线uv sync --locked和合成测试启动，不能承诺无私人数据即可得到旧checkpoint。清理工具缓存后原offline命令仅在缓存重新齐备时适用。环境.venv/.tools保留且仍忽略。独立失败/恢复快照与重建文件不硬链接合并，避免之后一处写入污染多个证据。

R1发布内容审阅：release_research_audit与release_engineering_audit先独立、再交叉确认无内容阻断；公开G5/G8聚合JSON与原结果深度相等、5项来源SHA匹配。该审阅不替代干净克隆/远端验证，完成状态见主控。

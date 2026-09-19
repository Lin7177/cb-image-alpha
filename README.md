# cb-image-alpha — paper replication branch

原论文美国股票实验的独立复现分支：官方图像/标签 → PaperCNN → 五成员训练 → OOS 指标与组合分析。中国可转债迁移版本保留在 `main`；本分支不把 CB 结果混入论文复现。

当前已完成官方公开包可得的 I20/R20 核心链。I5/I60 及其对应 weekly/quarterly 图像不在当前公开图像包中；组合分析已补齐为可审计的 Ret_month proxy 版本，并明确记录与作者 `next_month_ret` sidecar 的差异。严格 endpoint/leakage 结论仍未成立。

## 已有结果

- P3 五成员 I20/R20 训练已封存，使用官方公开图像/标签包和固定开发区间；checkpoint、恢复 probe、预测及资源账本均有本机证据。
- P4/r3 已完成一次受监督的已见 OOS 描述性重评：19 年、1,387,781 个保留样本；ensemble ROC AUC **0.537224**、accuracy **0.532823**、cross-entropy **0.690258**。这不是严格无泄漏确认性结果，endpoint_verified=false，也不恢复未见测试资格。
- I20/R20 的论文风格 pooled/按日期指标、10 分位 EW/VW、H-L 和 turnover 已补齐；收益使用 Ret_month proxy。其余八组配置因公开包缺少对应 image members 而数据阻断，不能用同一月频图像替代。

论文分支的运行证据位于本机 artifacts/paper-v1/（不提交）；配置覆盖清单见 artifacts/paper-v1/paper-scope-audit.json。转债迁移结果仍保留在 main 的 reports/cb-v1/，不与本分支结论混用。

## 从干净克隆开始

已验证平台为 Windows、Python 3.12、PyTorch 2.8.0 + CUDA 12.8。本项目的锁文件只支持 Windows；GPU 实验使用 RTX 5060 Laptop。安装 `uv` 后，在 PowerShell 执行：

```powershell
git clone https://github.com/Lin7177/cb-image-alpha.git
cd cb-image-alpha
uv sync --locked
$env:PYTHONPATH = 'src'
uv run --locked python -m pytest -q
uv run --locked python scripts/audit_release.py
```

首次安装需要网络及相应下载权限；仅在缓存齐全时使用 `--offline`。合成测试无需外部行情数据；可用 CUDA 时会执行相应合成检查。上述测试证明实现行为，不证明模型有预测优势。

## 复现边界与目录

| 目录 | 内容 |
|---|---|
| `src/cb_image_alpha/` | 数据解码、样本规则、渲染、CNN、训练与指标 |
| `configs/` | 冻结的转债数据、切分、训练和 G8 配置 |
| `scripts/` | 各 Gate 的构建、运行、验收及发布检查入口 |
| `tests/` | 独立手算、边界、梯度、保存恢复等合成测试 |
| `docs/` | 主控、契约、方法差异、开发与审计记录 |
| `reports/cb-v1/` | 可随 Git 分发的聚合结果与来源 SHA-256 |
| `artifacts/`（本机保留，不提交） | 完整索引、图像、checkpoint、预测、日志、manifest 与失败/恢复证据 |

私人 catalog 和历史 `artifacts/` 不随 Git 分发。**仅克隆源码不能恢复既有模型或核验全部本机证据**；必须合法取得相同数据及历史产物，并按 manifest 核对路径、字节摘要和环境。`python scripts/audit_release.py --local-evidence` 只核对本机 G5/G8 产物和冻结代码摘要，不训练或读取外部 catalog。

`scripts/research_g7.py`、`research_g8.py`、`recover_g8.py` 是绑定特定历史 run 的入口，不是任意数据通用训练命令；不要直接重新运行来覆盖或绕过既有次数限制。G6 重建命令与必要资料见[开发规范](docs/development.md)。根配置、源码与锁保留历史字节，`.gitattributes` 禁止 Git 自动换行转换。

外部数据仓库与 `Reference_repository/` 始终只读。论文 PDF、第三方代码、环境、数据和大体积研究产物不提交；第三方参考的再分发许可未获确认。此仓库不授予外部数据或论文材料的使用权。

## 文档入口

- [项目主控路线图 / Checklist](docs/plans/2026-09-10-foundation.md)：唯一当前状态与下一入口
- [系统架构与冻结契约](docs/architecture.md)
- [论文与实现迁移地图](docs/migration-map.md)
- [数据语义与限制](docs/data-audit.md)
- [开发、复现与发布规范](docs/development.md)
- [架构决策](docs/adr/0001-owned-minimal-pipeline.md)

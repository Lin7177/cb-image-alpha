# cb-image-alpha

中国可转债日频 OHLCV → I20 价格图像 → CNN → R5 收益方向预测的可复现研究实现。

**G0–G8 技术流程与开发比较已完成。预测有效性尚未得到支持。** 这是论文技术路线的转债迁移版本，不是原论文美股实验的完整复现。后续论文复现将使用独立分支，保留本版本及其负结果。

## 已有结果

- G5 首次 Final Test：AUC **0.498393**，log loss **0.714288**，差于训练常量概率基线 **0.696069**。Test 已查看，不再作为未见测试集。
- G8 三折时间顺序开发比较：Constant / U Logistic / Q Logistic / 原 CNN。U/Q 的主指标仅两折改善；量化差异小且方向混合；CNN 三折 log loss 均差于 Q。配对描述区间均跨零，不据此宣称 CV 无效。
- G9 未执行；建议停止当前 CNN 优化轴。

[可公开结果与来源摘要](reports/cb-v1/results.json)包含完整聚合指标，不含行级价格、标签、预测或私人目录。[架构文档](docs/architecture.md)解释比较方法及证据局限。

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

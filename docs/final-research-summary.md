# CB Image Alpha 最终研究总结（G0–P5）

日期：2026-09-19。当前研究轴在 P5 决策门后关闭；不进入后续阶段。

## 结论边界

**基本工程与方法流程已完成。** 项目完成了公开 I20/R20 数据获取与开发池审计、P2 样本索引/分区/训练集归一化、论文 PaperCNN 结构实现、生产训练器 CPU/CUDA 预检、故障注入、五成员真实训练、恢复核验和 checkpoint 封存。P3 经双 Agent 独立审计和交叉质疑后 PASS。

**原论文的完整预测评价复现未完成。** P4 评价协议虽冻结并执行了一次 OOS 读取，但实际耗时 7,992.609 秒，超过冻结的 7,200 秒上限；运行没有协议要求的 supervisor 或资源 ledger，最终双审未执行。因此 P4 权威状态是 BLOCKED。`results.json` 的 `COMPLETED_AWAITING_AUDIT` 只表示指标文件写出，不能覆盖 `blocked.json`。

**预测有效性未证实。** P4 运行产生的五成员指标和 ensemble 点估计只能作为一次失败运行的受限描述性诊断，不能作为正式 OOS 结果、确认性预测结论或严格无泄漏证据。`endpoint_verified=false`，`strict_leakage_free_claim=false` 始终成立。

## Gate 证据索引

- G0–G6：主控路线图及既有 Gate receipts；完成基本数据、渲染、CNN、切分、smoke、正式训练和复现路径核验。
- G7：零训练诊断完成，支持/削弱/不可识别假设台账保留在主控与研究文档。
- G8：Constant/U/Q/CNN 三折开发比较完成并保留负结果；未使用 Test 调参或收益回测。
- P1：官方数据恢复与开发池完整性完成；`Ret_month` 真实 endpoint 不可验证，跨 OOS 风险未排除；用户明确接受 Date proxy 的 paper-faithful 受限路线。
- P2：`paper-b-v1` PASS；793,019 原行对账至 778,507 合格行，五成员索引和 train-only 归一化冻结。
- P3：`paper-p3-training-v1` PASS；五成员无重试完成，最佳 epoch 为 6/8/4/4/12，验证预测、恢复 probe、资源和 checkpoint 链经双审核验。
- P4：协议已冻结，但执行 BLOCKED；OOS 已读取一次，不能再当作未见测试集。
- P5：只读决策门 PASS_DECISION_ONLY；确认 P4 结果只能保留为未验收描述性证据，关闭当前 OOS 评价轴。

## P4/P5 封存证据

最终诊断与决策记录：

- `artifacts/paper-v1/p4-oos-evaluation-v1/p4-diagnostic.json`
- `artifacts/paper-v1/p4-oos-evaluation-v1/diagnostic-integrity.json`
- `artifacts/paper-v1/p4-oos-evaluation-v1/p5-decision.json`
- `artifacts/paper-v1/p4-oos-evaluation-v1/blocked.json`
- `artifacts/paper-v1/p4-protocol-freeze.json`
- `artifacts/paper-v1/p3-paper-training-v1/p3-final-audit.json`

五个预测文件的存在、字节数、SHA、行数、唯一 ID 与有限概率已绑定在 `diagnostic-integrity.json`；该核验不重新读取 OOS 原始内容。

## 研究限制与停止决定

官方 `Ret_month` endpoint 未被公开数据可靠验证，Date proxy 可能存在跨 OOS 风险。P4 的一次读取已经消耗了未见测试边界；不得重读、重跑、调参、重训或回测。关闭当前 OOS 评价轴是研究决策，不是证明 CV 路线无效。

若未来重新评估，必须用户明确批准全新协议、预算、监督器和证据等级；不能把现有 P4 结果升级为正式结果。`scripts/paper_p4_evaluate.py` 是本次执行工作树中的未提交脚本，不属于已发布的干净可复现提交，不能据此宣称 P4 可复现。

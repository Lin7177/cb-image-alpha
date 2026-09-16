# 多 Agent 串行接力与合并审查协议

首要读者：主 agent（codex）。**codex 每次恢复工作，先读本文件**，确认有无待审查/合并的接力产物，再继续主控。

本文件是静态契约，不承载动态进度。当前"待审查队列"以 git 分支与 `artifacts/handoff/` 为准（见第 5 节）。

## 1. 背景与目的

主 agent（codex）额度有限、会断续中断。中断期间由接力 worker（另一 AI）续接确定性工作、产出证据；codex 恢复后审查、合并、继续裁决。本文件规定双方边界，保证：主线与主控权威只在 codex 手里，worker 不引入需要事后推翻的不确定内容。

## 2. 角色与权限

| 角色 | 允许 | 禁止 |
|---|---|---|
| 主 agent（codex） | 改 `main`、改主控 checklist 与 tracked 文档、下语义/设计结论、merge 分支 | — |
| 接力 worker | 跑确定性脚本、只读采集数据事实、聊天汇报、落 `work/*` 提案分支 | 不写 tracked 文档、不 merge、不下语义/设计结论、不脑补 repo 未写明的意图 |

## 3. 分工原则

唯一判据：**产出是否只由输入决定**。

- 确定性（跑已写脚本、只读采事实，结果 = f(代码+数据)）→ worker 可做。
- 判断/设计/语义结论（scale 判定、volume 单位、时区、复权口径、universe、切分、图像、CNN）→ 只有 codex。

codex 的稀缺资源是判断额度；worker 的价值是让 codex 恢复时面对现成证据，只花额度在裁决上。

## 4. 接力 worker 每轮边界

1. 先读主控 checklist，定位当前任务与下一步，不重定义计划。
2. 只做确定性项；结果只在聊天汇报，可选写 gitignored `artifacts/` 证据（须用户同意）。
3. 不写 tracked 文件、不 merge、不下结论；需要写 tracked 产物时，落 `work/<worker名>/<任务>` 分支作为提案，停在 merge 前。
4. 遇到 repo 未写明的决策 → 停下问用户，不自行假定。
5. 主线 `main` 与主控 checklist 始终不动。

## 5. codex 恢复时的固定动作（先于一切）

1. `git status --short`、`git log --oneline -n 10` —— 看有无新提交/未提交改动。
2. `git branch -a` —— 找 `work/*` 接力分支。
3. 对每个 `work/*` 分支 `git diff main...<branch>`，逐条裁决：合入 / 改后合 / 拒绝。
4. 看 `artifacts/handoff/` 最新交接文件，以及 `artifacts/environment/`、`artifacts/data_audit/` 证据——**非权威**，权威只有主控 checklist。
5. 回到主控 checklist 继续既定任务。

## 6. 分支与合并约定

- 主线：`main`。G0–G8执行时仓库尚无提交或remote；R1发布时建立 `main` 基线。当前分支、提交和远端以Git实际状态及主控发布记录为准。
- 接力分支：`work/<worker名>/<任务>`；worker 绝不直接提交 `main`。
- 合并：codex 执行 `git merge --no-ff`，review 后合入或拒绝。

## 7. 权威与证据层级

1. 主控 checklist（唯一权威，tracked，只 codex 改）。
2. tracked 代码/文档（分支提案，codex 合）。
3. gitignored `artifacts/`（确定性证据，非权威，时间戳命名）。

注意：`artifacts/`、`.venv/`、`.uv-cache/`、`__pycache__/` 均被 `.gitignore` 忽略，活在共享工作目录里、不受分支隔离。因此它们只作证据、不作权威；不同 worker 的 artifacts 靠时间戳命名区分，不靠分支。

## 8. 变更控制

本文件是治理契约，实质改动须用户确认。worker 不修改本文件与第 7 条第 1 层内容。

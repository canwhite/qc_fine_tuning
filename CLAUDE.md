## 新依赖：永远先读源码再写代码

**任何 `uv add` 的新库，动手写业务代码之前必须：**
1. 读包的源码（Python 在 `site-packages/` 或 `{package}/` 下）
2. 确认导出表、函数签名、调用方式、默认配置
3. 文档 ≠ 源码，文档可能是错的或过时的

**为什么这条规则存在：**
- `ppu-paddle-ocr` 库内 URL 拼写错误（ppdu vs ppu），文档没写
- Node 版 `PaddleOcrService` 需显式 `initialize()`，文档没写
- AI SDK v3→v6 迁移三次才读源码

**违反后的代价：** 调试绕路 1 小时+，远超过读源码的 10 分钟。

## Compact Instructions

When compressing, preserve in priority order:

1. Architecture decisions (NEVER summarize)
2. Modified files and their key changes
3. Current verification status (pass/fail)
4. Open TODOs and rollback notes
5. Tool outputs (can delete, keep pass/fail only)

## Verification

**Backend changes** (server routes, actors, storage):
1. `uv run build` — must pass
2. Kill any process on port 5555 (`lsof -ti:5555 | xargs kill -9`)
3. Start dev server: `uv run dev &` (background)
4. Wait ~3s, then curl all affected API endpoints to confirm correct response
5. Kill the dev server (`lsof -ti:5555 | xargs kill -9`)

**UI changes**:
1. `uv run build` — must pass
2. Capture before/after screenshots if visual

**Architecture/refactor changes**: do both backend and UI verification steps above.

**Definition of done**:
- Build passes
- Runtime verification passed (server starts, API responds correctly)
- No TODO left behind unless explicitly tracked

## 类型检测（Python）

**工具**: `mypy` (已内置于项目环境)

**强制要求**:
- 新增/修改的 Python 文件必须通过 `mypy --strict` 检查
- 关键函数必须标注返回类型（`-> type`）
- `Value` 类所有方法必须标注返回类型
- 避免变量重复类型注解（mypy strict 会报 `no-redef`）

**常见错误处理**:
- `Call to untyped function "X" in typed context` → 给函数/方法加 `-> type` 返回注解
- `Need type annotation for "x"` → 改为 `x: list[Type] = [...]`
- `Name "x" already defined` → 同作用域重复注解，用普通赋值替代

**运行方式**:
```bash
python -m mypy <file.py> --strict
```

## Debugging: framework primitives are not trustworthy

When a value looks wrong but the code that produces it appears correct, **instrument the framework boundary immediately**. Do not spend time reasoning about your own logic first — verify what the framework actually gives you.

### Case: Bun `params.id` returns `[http]-tcp:localhost:5555` (2026-06-17)

**Symptom**: JSONL files named `[http]-tcp:localhost:5555.jsonl` instead of `{slug}--{date}-{random}.jsonl`.

**Root cause**: Bun's `params.id` in route handlers returns the server's internal URL representation (`[http]-tcp:host:port`) instead of the actual URL path segment. This is a Bun framework bug.

**Why it took so long**:
1. Client-side logs showed the correct session ID being sent → assumed the problem was on the server
2. Server-side code looked correct — `getActor(params.id)` is trivially simple
3. Did not immediately suspect `params.id` itself could be wrong (framework primitive = trusted by default)
4. `[http]-tcp:...` format wasn't recognized as an internal Bun representation until server-side logging was added
5. The debug loop was loose — should have added `console.log("params.id:", params.id)` on the server as the very first step

**How to avoid next time**:
1. When data crosses a framework boundary (router → handler, middleware → handler, etc.), **log the raw value immediately** on the receiving side — don't trust that the framework passed it correctly
2. The format of wrong data is a clue: `[http]-tcp:...` looks like a URL/connection string → suspect the framework is passing its own internal state instead of the parsed parameter
3. Use `curl` to bypass the client entirely and confirm whether the server alone reproduces the bug → faster feedback loop
4. Don't spend more than 5 minutes reasoning about "why is my logic wrong" before adding instrumentation at the framework boundary

**Fix pattern**: Bypass the buggy framework API — parse the value directly from `new URL(req.url).pathname` instead of relying on `params.id`.

## SDK / Library 版本迁移：完整比对 checklist

当引入新版 SDK 或迁移库版本时，**必须先完整比对再写代码**，禁止边写边修。

### 必须核对的项

1. **导出表** — 新版本多了什么，少了什么？`Object.keys(exportedModule)` diff
2. **函数签名** — 参数名、返回值类型是否变了？查 `.d.ts` 声明
3. **调用方式** — 同步 vs 异步、流式处理、event callback 是否一致？
4. **数据格式** — 请求体结构、响应体结构、消息格式是否完全相同？
5. **破坏性变更** — 移除的 API 替代方案是什么？

### Case: AI SDK v3 → v6 迁移 (2026-06-17)

**症状**：ChatPanel 提交表单后页面刷新 → `handleSubmit is not a function` → `setInput is not a function` → 消息格式不匹配报错，连续三次迭代修复。

**根因**：只参考了 v3 文档就写代码，没有在 `node_modules/@ai-sdk/react/dist/index.js` 里核对 v6 实际返回的 hook 接口。

**为什么走了三遍**：
1. 第一遍：以为 `handleSubmit` 存在，只是没加 `preventDefault`
2. 第二遍：发现 `handleSubmit` 不在 v6 hook 返回值里，以为只是改名
3. 第三遍：发现 `input/setInput` 也不存在了，才去读源码

**正确的顺序应该是**：
1. `grep -n "return {" node_modules/@ai-sdk/react/dist/index.js` 找到 hook 返回值
2. 逐项核对：哪些 key 还在？哪些没了？新版本返回什么？
3. 查 `node_modules/ai/dist/index.d.ts` 的 `streamText` 返回类型，确认响应格式
4. 确认前端发送格式和后端期望格式是否匹配（v6 UIMessage parts vs streamText content）

**教训**：
- **永远先读源码确认，再写业务代码**。文档可能是旧版的。
- 对一个陌生版本升级，至少读 30 分钟源码再动手。
- 如果要改 3 遍以上才能跑通，说明第一次就没做够功课。

## Debugging: Docker 容器启动失败

**当容器立即退出且日志显示 usage/help 信息时**，说明容器没拿到正确参数，不是在"运行后退出"。

### Case: Milvus 容器退出，日志只显示 tini 用法 (2026-06-20)

**症状**：`docker ps` 显示容器 Exited，logs 只显示 tini 的帮助信息。

**根因**：Docker 镜像的 entrypoint 默认参数没有正确传递，导致 tini 收到了错误参数直接退出。

**错误做法**：
1. 只看 `docker logs` 的前几行，看到 tini usage 就以为"这个镜像有问题"
2. 反复 `docker run` 不带参数试
3. 没有查 `docker inspect` 的 ExitCode

**正确做法**：
1. `docker inspect <container> --format='{{.State.ExitCode}} {{.State.Error}}'` — 看退出码和错误信息
2. 退出码非 0 + 无具体 error = entrypoint 参数问题
3. 直接试显式命令：`docker run <image> milvus run standalone` 绕过 entrypoint 验证

**Why it took so long**：
- 日志开头是 tini usage，被误导为"镜像损坏"或"参数不对"
- 没意识到 tini 是 init 系统，它收到错误参数就直接退出了，根本没启动真正的 Milvus 进程
- 多次重试 `docker run` 不带参数的默认启动，而没有直接用 `milvus run standalone` 显式指定

**教训**：
- **容器日志开头是 usage/help 信息 = entrypoint 没拿到正确参数**，不是"运行后退出"
- 用 `docker inspect` 查 ExitCode 而不只是 `docker logs`
- **绕过有问题的默认启动命令，直接用进程名显式启动**：`docker run <image> <process> run <mode>`

## Debugging: 先看日志，不要轻易放弃

遇到问题时的第一反应必须是：**去看日志**，而不是反复尝试不同的启动方式或改代码。

**Case: Milvus 容器退出 (2026-06-20)**

错误流程：
1. 容器退出 → 只看 `docker logs` 前几行
2. 看到 tini usage → 误判为"镜像损坏"
3. 反复 `docker run` 试不同参数 → 不看日志，直接重试
4. 多次重试都没看完整日志 → 浪费时间

正确流程：
1. 容器退出 → `docker logs` 看完整输出
2. 发现是 tini usage → 说明 entrypoint 参数问题
3. `docker inspect` 查 ExitCode 确认
4. 试 `docker run <image> <进程名> run <模式>` 显式启动

**关键教训**：日志里有答案。不要跳过日志去猜原因。
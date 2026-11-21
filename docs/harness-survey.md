# Agent Harness 设计调研（2026-11-20）

> 目的：把 GitHub 主流 harness 与 PLE 官方 8 个模板的设计模式压成一张可执行清单，
> 用来判断我们自己的 `agent-harness` 该补什么、不该补什么。
>
> 取证方式：GitHub API 实测 star/许可证（2026-11-20）、官方 README/文档/论文原文、
> 本地 PLE 模板源码逐行阅读。未查证项一律标注。

---

## 一、结论先行

四个范式几乎覆盖了所有主流设计，取舍点不在"框架选哪个"，而在**四个机制的深度**：

| # | 机制 | 谁做得最好 | 为什么重要 |
|---|---|---|---|
| 1 | **验收驱动**（外部退出码裁定，不信 agent 自述） | SWE-bench / Terminal-Bench / plan-verify | Agent 会谎报 "tests pass"；只有 verifier 退出码能裁定 |
| 2 | **上下文压缩**（先截 tool 输出，再模型摘要） | Anthropic context engineering / custom-openai `compact()` | context rot：token 越多召回越差 |
| 3 | **观测设计**（错误写成"该怎么改"，截断必须声明） | SWE-agent ACI / Anthropic writing-tools | 小模型能不能正确调工具，取决于观测格式而非模型智力 |
| 4 | **健康恢复**（重试退避、停滞重启、断线 resume、收尾摘要） | minimal `recover()` / agy-gemini `STALL_SEC` | 长任务失败大多是"没跑完"不是"答错了" |

**对我们 harness 的判断**：机制 4 已完整（重试/cancel/usage/error 帧），机制 3 部分有（工具 schema + 路径逃逸防护 + 错误回灌），
**机制 1 和 2 是真缺口**——没有 verifier 概念，上下文只有"截断丢弃"没有"摘要压缩"。

---

## 二、PLE 官方 8 模板（本地源码逐行读）

| 模板 | serve.py 行数 | 核心设计 | 值得借鉴的具体机制 |
|---|---|---|---|
| `custom-openai` / `local-model` | 1800 | 全功能参考 | `compact()`：先截旧 tool 输出（保留 `KEEP_RECENT_RESULTS`），不够再调模型摘要；`finish()`：时间到前发无工具收尾请求；`nudges`/`head`：输出被 `finish=length` 截断时续写 |
| `minimal` | 373 | 最小循环 | `WRAP_UP_SEC=240` 提前 240s 要摘要；`MAX_NUDGES=3` 连续空回复才停；`RETRY_WAITS=(5,15,30,60,120)`；`shorten()` 把旧一半 tool 结果截到 `CUT_KEEP=1000` 字符 |
| `plan-verify` | 518 | 三角色 | `PLANNER`（只读工具写计划）→ `WORKER`（全工具执行）→ `VERIFIER`（只读，回 PASS 才算完）；`MAX_FIX_ROUNDS` 反馈环 |
| `agy-gemini` | 571 | MCP 优先 | `tools: []` 内置工具全禁，全靠 MCP；`STALL_SEC=2400` 无输出即停；`MAX_RUNS=4` 断线/限流/停滞在同会话 resume |
| `hermes-opus` | 366 | terminal 后端 | 文件工具走 terminal（`terminal: {backend: task, cwd, timeout:180}`），不单独暴露 read/write |
| `pi-opus` | 21 + `harness.py` | TS harness | serve.py 只做 `import harness`，逻辑全在 TS |
| `node-protocol` | `serve.mjs` | Node 实现 | 协议同构的跨语言验证 |

**共同骨架**（8 个模板全都遵守）：
- `taskbridge` 解耦：harness 不直接碰容器，所有 exec/read/write 走 relay 请求
- `cache_control` 标记 system + 最新消息（Claude 前缀缓存，省 token）
- 时间感知：`session.time_left()` 决定 wrap-up 时机，不是固定步数

---

## 三、GitHub 主流框架（4+4 个，实测 star/许可证 2026-11-20）

### 编码 agent 类

| 框架 | Star | 许可 | 主循环 | 最值得借鉴 |
|---|---|---|---|---|
| **SWE-agent** `SWE-agent/SWE-agent` | 20,838（2万级） | MIT | ReAct（ACI 定制命令） | ① **linter 前置拦截**：语法错误的 edit 不写入状态；② 100 行窗口视图 + 空输出规范提示 |
| **OpenHands** `OpenHands/OpenHands` | 93,898（9万级） | MIT | 事件驱动 ReAct | ① **append-only EventStream 单一真相源**，全组件不可变、可回放；② stderr/exit_code 结构化回灌 |
| **smolagents** `huggingface/smolagents` | 30,458（3万级） | Apache-2.0 | code-as-action | ① **动作即代码**，一段代码一次执行多工具调用；② 错误分类 + traceback 回灌自修复 |
| **LangGraph** `langchain-ai/langgraph` | 45,190（4万级） | MIT | graph 状态机 | ① **superstep 事务性**：并行分支要么全生效要么全不生效；② checkpoint + time travel |

### SDK / 轻量类

| 框架 | 许可 | 定位 | 最值得借鉴 |
|---|---|---|---|
| **OpenAI Agents SDK** | MIT | 多 agent 编排 | ① **handoff-as-tool**：`transfer_to_<agent>` 是普通工具，参数 schema 由目标 agent 的 `input_type` 生成；② **tripwire guardrail** + blocking/parallel 两模式 |
| **pydantic-ai** | MIT | 类型驱动 SDK | ① **签名+docstring 自动生成 tool schema**，参数在代码运行前校验；② **capabilities 组合式中间件**（可整用可拆用）；③ `RepairToolArguments` 自动修实参 |
| **Letta / MemGPT** | Apache-2.0 | 有状态 agent | ① **分层记忆** core（常驻）/ recall（可搜历史）/ archival（向量库）；② agent 用 tool call 自编辑记忆；③ `heartbeat` 强制多步完成 |

### 设计原则（权威来源）

| 来源 | 关键原则 |
|---|---|
| Anthropic《Building Effective Agents》(2024-12-19) | **simplicity**：从最简单可工作方案开始；**ACI**：像投入 HCI 一样投入工具文档；**poka-yoke**：强制绝对路径消灭相对路径错误 |
| Anthropic《Effective context engineering》(2025-09-29) | **context rot**：token 越多召回越差；**just-in-time context**：上下文只放轻量标识符，按需加载；长任务三技术 = compaction / NOTES.md / sub-agent |
| Anthropic《Writing effective tools》 | 少而精；namespacing 划边界；**concise/detailed 两档**让 agent 控制详略；截断必须声明；错误写成可行动指令 |
| SWE-bench (ICLR 2024, arXiv:2310.06770) | 验收 = patch 应用 + 容器跑仓库测试；**失败分型**（patch/测试/超时/基础设施）；run_id 缓存键防假通过 |
| Terminal-Bench (ICLR 2026, arXiv:2601.11868) | 每任务自带 **verifier + oracle**；分数 = task 完成率；没有 verifier 的任务不进 benchmark |
| LangChain《Context Engineering》(2025-07-02) | write / select / compress / isolate 四类操作；四问题 = poisoning / distraction / confusion / clash |

---

## 四、对 `agent-harness` 的具体改进清单（按 ROI 排序）

### P0 — 真缺口，直接对应"agent 会谎报成功"

**1. 验收钩子（verify command）**
- 现状：`agent.run()` 返回 `finished=True` 就结束，对"是否真的完成任务"零判断。
- 方案：`Config.verify_command: str = ""`。主循环结束前若设置了它，就执行一遍；退出码非 0 则把 stderr 回灌给模型再跑（限 `max_verify_rounds`）。
- 借鉴：plan-verify 的 `VERIFIER` 阶段 + SWE-bench"外部测试退出码裁定"。
- 价值：这是`examples/run_demo.py` 里"独立 pytest 识破模型谎报"的机制化版本。

**2. 上下文摘要压缩**
- 现状：`context.py:fit()` 只做"截 tool 输出 + 丢旧消息"，丢了就没了。
- 方案：两级——① 先截旧 tool 结果到 `KEEP_RECENT_RESULTS` 条之外（已有 `max_tool_result_chars`）；② 仍超预算时调一次模型把旧对话压成 `NOTES` 摘要消息。
- 借鉴：custom-openai `compact()`（1661 行起）+ minimal `shorten()`（251 行起）。
- 价值：让长任务不因"丢历史"而失忆。

### P1 — 强增益

**3. `edit` 工具（精确替换）**
- 现状：只有 `write_file` 整文件覆盖——qwen 覆盖丢行的根因之一。
- 方案：`edit(path, old_string, new_string)`，要求 `old_string` 唯一匹配，否则报错回灌。
- 借鉴：SWE-agent `edit` + linter 前置拦截。
- 价值：直接降低小模型丢行/写坏文件的概率。

**4. 观测格式升级**
- 现状：工具返回裸字符串，错误是 `[tool error: ...]`。
- 方案：错误响应写"该怎么改 + 正确示例"；截断时明确声明"已截断 + 如何取剩余"。
- 借鉴：Anthropic writing-tools + SWE-agent 空输出规范提示。

### P2 — 按需

**5. `finish()` 收尾摘要**：时间/步数接近上限时，发一次无工具请求要摘要，避免"跑满 20 步什么都没留下"。借鉴 minimal `WRAP_UP`。
**6. 停滞检测**：连续 N 步无新 tool 调用或输出重复 → 提前收尾。借鉴 agy-gemini `STALL_SEC`。
**7. `concise`/`detailed` observation 档位**：省 ~2/3 token。借鉴 Anthropic writing-tools。

### 明确不做

- **多 agent / sub-agent 编排**：当前单 agent 骨架，做这个违背 Simplicity First。
- **MCP 集成**：PLE 协议里有 `mcp_servers`，但通用场景用不上；要做也是独立模块。
- **代码即动作（smolagents 式）**：需要沙箱执行 Python，本机已有 bash 工具覆盖同类需求。
- **分层记忆（Letta 式）**：超出一个轻量 harness 的职责。

---

## 五、来源清单

- `https://github.com/SWE-agent/SWE-agent`（MIT，20,838 star）
- `https://github.com/OpenHands/OpenHands`（MIT，93,898 star）
- `https://github.com/huggingface/smolagents`（Apache-2.0，30,458 star）
- `https://github.com/langchain-ai/langgraph`（MIT，45,190 star）
- `https://github.com/openai/openai-agents-python`（MIT）
- `https://github.com/pydantic/pydantic-ai`（MIT）
- `https://github.com/letta-ai/letta`（Apache-2.0）
- `https://www.anthropic.com/research/building-effective-agents`（2024-12-19）
- `https://www.anthropic.com/research/effective-context-engineering-for-ai-agents`（2025-09-29）
- `https://www.anthropic.com/research/writing-tools-for-agents`
- `https://github.com/SWE-bench/SWE-bench`（arXiv:2310.06770）
- `https://github.com/laude-institute/terminal-bench`（arXiv:2601.11868）
- `https://www.langchain.com/blog/context-engineering-for-agents`（2025-07-02）
- 本地：`ple-competition/sample_submissions/{minimal,plan-verify,custom-openai,agy-gemini,hermes-opus,pi-opus,node-protocol,local-model}/`

未查证项：SWE-agent summarizer 触发阈值；smolagents 上下文压缩策略；OpenAI `@function_tool` 装饰器名（文档页超时）；Letta 具体 memory tool 函数名。

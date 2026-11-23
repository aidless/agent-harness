# agent-harness

一个最小但完整的 **agent harness**：给定一个任务，模型自己决定调用工具（跑命令、读文件、写文件），在循环里推进直到给出最终答案。不绑定任何厂商 SDK——只通过 `httpx` 直连一个 OpenAI 兼容的 `/chat/completions` 端点，因此本地 ollama、vLLM、SGLang、或任何 OpenAI 兼容网关都能直接跑。

参赛只是顺带：这套代码对齐了 [Google Physical-Engineering's Last Exam](https://www.kaggle.com/competitions/physical-engineerings-last-exam) 的 harness 形态（工具循环 + 上下文压缩 + 轨迹记录），但本身完全独立，可拿去干别的事。

## 设计目标

| 目标 | 做法 |
|---|---|
| 少依赖 | 只有 `httpx` + `PyYAML`，不用 openai SDK，避开 SDK 版本坑 |
| 可换模型 | 一个 `base_url` + `model` 就能切换 ollama / vLLM / 云 API |
| 能真跑 | 本地 ollama 实测：写文件/读回、WebSocket 协议闭环通过；多步 bugfix 受限于小模型能力（见下） |
| 可观测 | 每一步 LLM 回复、工具调用、结果都落 JSONL 轨迹 |
| 防抽风 | 小模型在 temperature=0 会一次吐几十个重复 tool_call，harness 自动去重 |

## 快速开始

```bash
# 0) 安装（核心只需 httpx + PyYAML；serve 模式另装 extra）
pip install .                 # CLI：agent-harness task ...
pip install '.[serve]'        # 额外启用 WebSocket serve 模式

# 1) 起一个 OpenAI 兼容端点（示例：本地 ollama）
ollama serve &
ollama pull qwen2.5:7b

# 2) 准备配置
cp config.example.yaml config.yaml
#   按需改 base_url / model

# 3) 跑一个任务
python -m agent_harness task "在当前目录创建 hello.py 打印 hello，然后运行它"

# 4) 指定工作目录、限制步数、落轨迹
python -m agent_harness task \
  --config config.yaml \
  --workdir /tmp/my-work \
  --max-steps 12 \
  --trace /tmp/my-work/trace.jsonl \
  "修复 src/ 下的 bug，让 pytest 全绿"
```

输出最终答案；`--json` 打印结构化结果（answer/steps/tool_calls/finished/error）。

### WebSocket 服务（PLE v1 协议）

把 harness 作为一个 WebSocket server 跑起来，评测器发 `task` 帧、harness 把
工具请求（`exec` / `read_file` / `write_file`）发回评测器执行，结束发 `done`：

```bash
PORT=8765 MODEL_NAME=qwen2.5:7b \
  MODEL_BASE_URL=http://127.0.0.1:11434/v1 \
  python -m agent_harness.serve
```

`MODEL_NAME` / `MODEL_BASE_URL` / `MODEL_API_KEY` 由评测器注入（也认
`OPENAI_BASE_URL` / `OPENAI_API_KEY`）；没有则用 `AGENT_*` / 配置默认值。
收到 `cancel` 后工具立即失败返回，在途的模型调用也会被中断（`LLM.abort()`），
循环抱着已有结果收尾。模型调用会发 `usage` 帧（input/output/cache_read tokens）；
主循环出现错误时发 `error` 帧（已写入的文件仍参与评分），正常结束发 `done`。
`tests/serve_smoke.py` 是一个 mock 评测器，可本地端到端自测这一层：

```bash
python tests/serve_smoke.py --workdir /tmp/serve-smoke \
  "Create hello.txt with 'hi' then read it back"
```

### 多步任务演示

```bash
python examples/run_demo.py --model qwen3:8b \
  --base-url http://127.0.0.1:11434/v1 --workdir /tmp/demo-bugfix
```

演示脚本复制一个带 bug 的小项目，让 agent 跑 pytest → 定位 → 改代码 →
复测，然后**在 harness 之外独立跑一次 pytest** 验证。退出码 0 当且仅当
独立验证通过——不看 agent 自己说的 "tests pass"。

### 验收关口（verify_command）

不让模型自己宣布成功：给 `verify_command` 配一条命令（通常是跑测试），
harness 在每次最终答复前真的执行它，退出码非 0 就把输出回灌给模型再修一轮。
`examples/verify_demo.py` 用脚本化 LLM 演示了完整闭环——模型先谎报 "done"，
被真 pytest 打回、回灌失败输出、改对代码、再验通过：

```bash
python examples/verify_demo.py
```

退出码来自 shell 哨兵（`cmd ; echo __AH_VERIFY_EXIT__=$?`），所以无论工具把
stdout/stderr 怎么合并，harness 都能可靠判定通过与否。配置示例见
`config.example.yaml`。

优先级：dataclass 默认值 → YAML 文件 → `AGENT_*` 环境变量（环境变量最高）。

| 字段 | 默认 | 说明 |
|---|---|---|
| `base_url` | `http://172.17.0.1:11435/v1` | OpenAI 兼容端点（含 `/v1`） |
| `model` | `qwen3:8b` | 模型名，要与端点 `/v1/models` 里的名字一致 |
| `api_key` | `local` | 发到 `Authorization: Bearer` |
| `max_steps` | `20` | 最多多少轮 agent 循环 |
| `max_tokens` | `4096` | 每次 LLM 回复的最大 token |
| `temperature` | `0.0` | 采样温度 |
| `timeout` | `300` | 单次 HTTP 请求超时（秒） |
| `max_retries` | `5` | 瞬态失败（429/5xx/网络错）的**重试次数**（首次之外） |
| `retry_base_delay` | `1.0` | 第一次重试前的退避基数（秒，指数增长 + 抖动） |
| `retry_max_delay` | `30.0` | 单次退避上限，也用于 `Retry-After` 封顶 |
| `context_window` | `32768` | 送进模型的预算，超了先截 tool 输出再丢旧消息（字符或 token，见下） |
| `max_tool_result_chars` | `12000` | 单个 tool 结果最多保留的字符数 |
| `count_tokens` | `false` | `true` 时 `context_window` 按**精确 token** 计（需 `pip install tiktoken`） |
| `verify_command` | `""` | 非空则作为**验收关口**：模型给最终答复前先跑它，退出码非 0 就把输出回灌让模型重试 |
| `max_verify_rounds` | `2` | 验收失败时最多回灌重试几轮 |
| `verify_timeout` | `120` | 验收命令超时（秒） |
| `compact` | `false` | `true` 启用**两级上下文压缩**：历史将被丢弃前先用模型摘要保留关键事实（多一次 LLM 调用） |
| `compact_after_messages` | `12` | 消息数达到该值才开始考虑压缩 |
| `compact_keep_recent` | `4` | 压缩时保留最近多少条消息不摘要 |
| `workdir` | `.` | 工具（bash/read_file/write_file）的工作目录 |
| `trace_path` | `""` | 非空则把轨迹写到这里（JSONL） |
| `checkpoint_path` | `""` | 非空则**每步**把 messages + 计数器原子落盘到该 JSON（崩溃最多丢一步） |
| `resume` | `false` | `true` 且 checkpoint 存在时，从断点续跑；任务不匹配或文件损坏则从新开始 |

环境变量覆盖示例：`AGENT_MODEL=qwen2.5:7b AGENT_MAX_STEPS=10 python -m agent_harness task "..."`。

## 内置工具

| 工具 | 参数 | 行为 |
|---|---|---|
| `bash` | `command`, 可选 `timeout` | 在 `workdir` 下跑 shell 命令，返回 stdout+stderr+退出码 |
| `read_file` | `path` | 读文件，超限截断 |
| `write_file` | `path`, `content` | 写文件（自动建父目录） |

所有路径操作限制在 `workdir` 内，`..` 逃逸会被拒绝。

## 结构

```
agent_harness/
  config.py    配置加载（dataclass / yaml / env 三层）
  llm.py       OpenAI 兼容 /chat/completions 客户端（httpx，非流式）
  tools.py     工具 schema + 实现（bash / read_file / write_file）
  context.py   上下文：超限先截 tool 输出 → 丢最旧消息；compact 再补一级"模型摘要"
  agent.py     Agent 主循环：LLM → tool_calls → 执行 → 回灌 → 验收关口 → 下一步
  trace.py     JSONL 轨迹记录
  serve.py     PLE v1 WebSocket 服务端（工具在远端容器执行）
  __main__.py  CLI 入口
config.example.yaml
tests/                     单元测试（79 个，纯 mock，不依赖模型端点）
tests/serve_smoke.py       mock 评测器，端到端自测 serve 层
tests/test_multistep_e2e.py  脚本化 LLM 驱动的多步 bugfix 端到端测试
examples/run_demo.py       多步 bugfix 演示（用真实 ollama 模型）
examples/scripted_demo.py  同一任务，用脚本化 LLM 驱动（确定性，证明主循环正确）
examples/verify_demo.py    验收关口演示：模型谎报→测试打回→修复→再验通过
examples/bugfix-project/   演示用 fixture 项目
docs/harness-survey.md     GitHub 主流 harness 与 PLE 模板设计调研
```

## 已知行为与取舍

- **重复 tool_call 去重**：同一轮里同名同参数的调用只执行一次，其余回 `[duplicate call skipped]`，且每轮最多执行 8 个工具。qwen2.5:7b 在复杂 prompt + `temperature=0` 下会一次吐 46 个一模一样的 `read_file`，这不是 harness 的 bug，是模型行为；去重后实际执行从 49 次降到 4 次，结果不变。
- **瞬态失败自动重试**：`429 / 408 / 409 / 425 / 5xx` 与网络错误按指数退避重试（默认 5 次，1s 起、30s 封顶、50–100% 抖动），优先遵守 `Retry-After`；`400/401/403/404` 视为永久错误直接抛出。重试等待可被 `abort()`（serve 收到 `cancel` 帧）立即打断，不会卡住任务超时。
- **上下文管理默认字符级，可选精确 token**：默认零依赖、按字符计预算；设 `count_tokens: true`（或 `AGENT_COUNT_TOKENS=1`）即用 `tiktoken` 精确计数，只改"保留/丢弃"的分界，不改输出形状（tool 输出截断仍按字符，属粗粒度安全阀）。
- **非流式**：一次拿完整回复，简单可靠；要低延迟首字可改流式。
- **单轮无并发**：一个 task 一个 agent，没有多 agent / 并行工具；需要可扩展时在这个骨架上加。
- **模型能力决定多步任务的成败**：harness 只提供正确的工具循环与验证，不会替模型补齐推理。实测本机三个小模型在多步 bugfix 上全败：qwen2.5:7b 把操作写成伪代码文本、写文件时丢掉未改的行；qwen2.5-coder:14b 把 tool_call 塞进 content 字符串；qwen3:8b 写出语法错误的文件。同一个 qwen2.5:7b 在单步的 serve smoke 任务上却通过。这是模型侧瓶颈，不是 harness 缺陷。
- **多步主循环已用脚本化 LLM 证明正确**：`examples/scripted_demo.py` 用一个进程内假端点按固定顺序返回 tool_calls（读文件 → 整文件覆盖修复 → 跑 pytest → 收尾），驱动真实 Agent 跑完 4 步 3 次工具调用，独立 pytest `3 passed`。这把"多步能不能成"与本机模型能力完全解耦——挂掉的是小模型，不是 harness。同一验证固化为回归测试 `tests/test_multistep_e2e.py`。要验证换更强模型的效果，改 `examples/run_demo.py` 的 `--model`/`--base-url` 即可。
- **验收关口防谎报**：设 `verify_command` 后，模型每次给出最终答复前 harness 都真跑一遍该命令；退出码非 0 就把输出回灌给模型再修，最多 `max_verify_rounds` 轮。退出码用 shell 哨兵 `; echo __AH_VERIFY_EXIT__=$?` 从合并输出里抽，不受 stdout/stderr 混合影响。`max_steps` 用尽时若验收通过仍判成功。`examples/verify_demo.py` 端到端演示了"谎报 done→pytest 打回→修复→再验通过"，`verify_rounds=2`、独立 pytest 3 passed。
- **两级上下文压缩**：`compact: true` 时，历史将被 `fit()` 丢弃前先用模型摘要成一条短 note（保留文件路径、失败、待办），系统提示与最近 `compact_keep_recent` 条原样保留；摘要调用失败自动回退到普通 `fit()`。这补上了"只截断丢弃"会丢失长任务历史的问题，代价是每次压缩多一次 LLM 调用。默认关闭。
- **断点续跑（checkpoint/resume）**：设 `checkpoint_path` 后，每步把 `messages` + 计数器（`steps`/`tool_calls`/`verify_rounds`）原子落盘（临时文件 + `os.replace`，崩溃不留半截文件），崩溃最多丢一步；下次带 `--resume`（或 `resume: true`）同一任务即接着跑，已执行的工具不重放、已完成的回合不重复调模型。checkpoint 里 `task` 不匹配或文件损坏时自动从头开始。CLI 用 `agent-harness task ... --checkpoint run.json --resume`。
- **系统提示词要短**：qwen2.5:7b 在 `temperature=0` 下，若 system prompt 含 "Verify your work by running the project's tests ... and fix whatever fails" 一类句子，会直接空回复（`content=''`, `tool_calls=0`），5/5 可复现；删掉该句即恢复。默认提示词已避开。

## 与竞赛的关系

这套 harness 的形状（工具循环 + 上下文压缩 + 轨迹）对标 PLE 竞赛要求。`agent_harness/serve.py` 已实现 PLE v1 WebSocket 协议（`task` 帧进、`exec`/`read_file`/`write_file` 请求出、`log`/`usage` 单向帧、`done`/`error` 收尾、`cancel` 中断在途请求），并用 `tests/serve_smoke.py` 端到端跑通。不参赛时，CLI（`python -m agent_harness task`）就是一个能用的通用 agent harness。

## License

MIT。
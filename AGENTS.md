# AGENTS.md

本文件是 `Agent_System` 仓库内后续开发、重构、调试和 Agent 协作的统一工作约定。

目标不是维护一个“通用聊天壳”，而是把这个项目逐步演进成一个 **以学术检索为示例的 Agent 调试/演示平台**。后续所有架构决策、界面设计、Context 管理和运行链路优化，都应优先服务这个目标。

## 1. 项目定位

### 1.1 当前定位

本项目当前保留“学术检索 Agent”这个真实业务场景，但后续重构的主目标不是单纯提升聊天体验，而是构建一个适合学习和演示的 Agent 平台。

平台需要让使用者清楚看见：

`用户问题 -> Agent 决策 -> 工具调用 -> 状态更新 -> Context 装配 -> 最终回答`

### 1.2 第一阶段目标

第一阶段重构应围绕以下目标展开：

- 把项目从“消息驱动的聊天系统”升级为“事件 + 状态驱动的 Agent 演示平台”
- 把 `Context` 从“历史消息注入”升级为“分层上下文装配”
- 把前端从“聊天页 + 调试面板”升级为“执行过程优先的 Research Workbench”
- 保留当前学术检索示例场景，不在第一阶段去掉论文检索能力

### 1.3 非目标

第一阶段不追求以下事项：

- 不做通用多 Agent 平台
- 不暴露原始思维链
- 不为移动端优先设计
- 不引入新的数据库或基础设施
- 不改动现有 embedding 服务、Milvus 检索结构和 Redis 基础依赖

## 2. 默认环境约定

### 2.1 Python 环境

默认使用本机已有的 `jzy` Python 环境执行所有 Python 代码：

`D:\Anaconda\envs\jzy\python.exe`

除非某个任务明确要求其他解释器，否则：

- 启动脚本
- 调试脚本
- 数据处理脚本
- 临时验证命令

都默认使用这个 Python 路径。

### 2.2 运行依赖

默认运行环境依赖如下：

- Redis：`localhost:6379`
- Milvus：`localhost:19530`
- 前端：Vite
- 后端：FastAPI + Uvicorn
- Embedding 稠密向量服务：`http://localhost:54331/v1`
- Embedding 稀疏向量服务：`http://localhost:54332/v1`

这些依赖在第一阶段重构中默认保持不变。

### 2.3 调试交接约定

后续每次任务完成后，默认要把本地调试环境保持在“可立即打开浏览器继续检查”的状态。

最低要求：

- 后端服务保持运行：`http://127.0.0.1:8000`
- 如本轮涉及前端改动，前端开发服务也应保持运行：`http://127.0.0.1:5173`

这条约定的目的不是“启动过一次就算完成”，而是让用户在任务结束后可以立刻：

- 打开浏览器复现
- 继续点界面找问题
- 观察 SSE、工具调用、状态变化和上下文装配

如果由于端口占用、进程崩溃或权限问题未能保持服务在线，必须在最终说明里明确指出，而不是默认视为已交付。

## 3. 模型与密钥约定

### 3.1 主对话模型

后续默认主对话模型不再使用原先本地 `gpt-5.1` 对应的主模型链路，而是切换为：

- 模型名：`MiniMax-M2.7`
- Base URL：`https://api.minimaxi.com/v1`

当前 LangChain/OpenAI 兼容链路不要传 `extra_body={"reasoning_split": True}`，该参数会触发 MiniMax `invalid chat setting(2013)`。

由于不启用 `reasoning_split` 时模型可能把 `<think>...</think>` 混入普通文本流，后端 SSE 层不能把这类原始思维标签混在最终回答文本里直接展示。

允许后续把原始 `<think>` 内容作为单独的调试字段捕获，例如挂在某一次 `Model Call` 的 `reasoning_trace` / `raw_think` 上。但它必须遵守渐进式披露：

- 默认不在主回答区、时间线卡片或第一屏展示
- 只能在用户点击具体某一次模型调用后，通过深层窗口或抽屉查看
- UI 必须明确标识这是原始调试信息，不等同于最终答案或结构化状态
- 如果后端尚未提供独立字段，前端不得从最终回答文本里强行拼接或猜测 think 内容

MiniMax 兼容接口还有一个已验证限制：同一次 Chat Completions 请求中不能出现多条 `system` message，否则会返回 `invalid chat setting(2013)`。因此：

- Agent 自带的 `SYSTEM_PROMPT` 是唯一真正的 system message
- 历史摘要、压缩上下文、状态说明不能再以 `SystemMessage` 形式注入模型请求
- 如需注入历史摘要，应转换为普通上下文说明消息，或通过 Context Block 展示和装配记录表达

### 3.2 Embedding 模型

本次重构中，以下链路保持不变：

- 稠密向量 embedding 服务
- 稀疏向量 embedding 服务
- Milvus 检索逻辑
- 检索工具对 embedding 的依赖方式

### 3.3 密钥约定

本机存在私有凭证配置，但：

- 不要把任何明文 API Key 写入仓库
- 不要把任何明文 API Key 写入 `AGENTS.md`
- 不要把任何明文 API Key 写入测试脚本、README 或提交记录

代码和配置应优先通过环境变量读取密钥。后续如需统一，优先兼容以下变量思路：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`

如新增 MiniMax 专用变量，也应通过环境变量读取，不要硬编码。

补充本地运行约定：

- 允许在仓库根目录使用本地忽略文件 `.env.local` 保存本机运行所需的私有配置
- `.env.local` 必须保持在 `.gitignore` 中，不作为仓库提交内容
- 当前默认加载顺序为：
  - 显式指定的 `AGENT_SYSTEM_ENV_FILE`
  - 仓库根目录 `.env.local`
  - 仓库根目录 `.env`
- 后端重启后应能自动从本地文件恢复 `MiniMax` 配置，不依赖每次手动注入 PowerShell 环境变量

当前本地文件至少应支持：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`

### 3.4 历史模型说明

仓库中原有的本地主模型脚本和 `gpt-5.1` 相关说明，可视为历史实现或 legacy reference。在后续重构中：

- 可以保留作为参考
- 但不应继续把它当作当前默认主模型方案

## 4. 架构重构总原则

### 4.1 从 message 驱动升级为 event + state 驱动

当前系统大量依赖消息历史来承载运行信息，这种方式不适合做可解释的 Agent 调试平台。后续重构应把系统主结构升级为：

- `Message`：用于表达用户输入和最终回答
- `Event`：用于记录运行过程中发生的关键行为
- `Derived State`：用于记录当前任务状态、工作记忆和结构化上下文
- `Context Block`：用于表达上下文来源和装配单元
- `Assembly Record`：用于记录一次模型调用前的上下文选择和装配结果

### 4.2 规则优先，不依赖黑盒选择

第一阶段的 Context 装配采用“规则优先”策略，而不是把所有上下文选择权交给模型。

后续实现应优先做到：

- 行为稳定
- 可复现
- 可解释
- 适合教学

允许后续扩展“模型辅助选择”，但不是第一阶段主线。

### 4.3 原始思维过程的展示边界

平台第一信息层要展示的是结构化运行过程，而不是模型自由文本思维链。原始 think 过程可以作为深层调试材料存在，但不能成为默认阅读对象。

前后端都应优先展示以下类型的信息：

- 当前任务目标
- 是否需要外部证据
- 选择了哪个工具
- 工具观察结果是什么
- 哪些状态发生了变化
- 哪些上下文块被纳入或剔除
- 最终回答如何生成

原始 think 过程的展示规则：

- 默认折叠，不进入首页主线
- 必须绑定到某一次明确的 `Model Call`
- 必须通过点击、弹窗、抽屉等深层交互查看完整内容
- 默认摘要层优先展示 `Decision Note` / `Reasoning Summary`
- 完整原文只用于调试和教学拆解，不用于替代状态、证据或最终回答

## 5. Context 重构方法

### 5.1 Context 不再等于历史消息

后续实现中，`Context` 不能再简单理解为“摘要 + 最近消息”。需要把它拆成用途明确的上下文块。

建议的核心上下文块类型：

- `system_rules`
- `task_goal`
- `current_user_request`
- `working_memory`
- `confirmed_facts`
- `active_constraints`
- `recent_dialogue`
- `tool_evidence`
- `historical_summary`

每个 Context Block 至少应具备以下元数据：

- `source`
- `priority`
- `freshness`
- `estimated_tokens`
- `selected`
- `drop_reason`

### 5.2 Context 装配过程

每次模型调用前，不应直接注入历史消息，而应经过显式的装配过程：

1. 收集候选上下文块
2. 基于规则、优先级和预算做筛选
3. 记录哪些块被选中
4. 记录哪些块被剔除以及原因
5. 生成最终 prompt/message payload
6. 写入 `Assembly Record`

这个过程必须可回放、可展示、可调试。

### 5.3 压缩策略

旧方案是“消息多了以后统一摘要”。新方案应改为“分层归纳”：

- `recent_dialogue`：保留短窗口原文
- `historical_dialogue`：压缩成摘要
- `tool_evidence`：压缩成结构化结论和引用索引
- `state`：不压成自由文本，而是做字段级更新

压缩目标不是单纯变短，而是让 Context 更适合装配和回放。

## 6. 状态模型与回放

### 6.1 Event Log

后续系统需要保留按时间顺序记录的事件流。建议事件至少覆盖：

- `user_message_received`
- `phase_changed`
- `tool_called`
- `tool_result_received`
- `state_updated`
- `context_block_selected`
- `context_block_dropped`
- `assembly_completed`
- `assistant_answer_streamed`
- `assistant_answer_completed`

### 6.2 Derived State

当前状态视图至少应包含：

- `current_goal`
- `working_memory`
- `confirmed_facts`
- `active_constraints`
- `open_loops`
- `active_evidence_refs`
- `recent_context_summary`

这个状态不是简单消息列表，而是从事件归约得出的结果。

### 6.3 快照策略

平台需要支持回放，因此应采用：

- `事件流 + 关键事件快照`

关键事件后应允许保存状态快照，例如：

- 工具调用前后
- 工具结果返回后
- 状态更新后
- Context 装配完成后
- 一轮回答完成后

第一阶段不要求对每个 token 做快照。

### 6.4 Redis 角色

第一阶段继续使用 Redis，不引入新持久层。Redis 的角色应从“简单消息缓存”升级为：

- 事件流存储
- 派生状态缓存
- Context Block 缓存
- Assembly Record 缓存
- Tool/Evidence 缓存

## 7. 前端工作台方向

### 7.1 总体方向

前端不再以“普通聊天页面”为目标，而应演进为：

- 桌面优先
- 执行过程优先
- 结构清晰
- 可渐进披露

推荐视觉方向：`Research Workbench`

### 7.2 首页叙事

首页的第一信息层应优先展示执行过程，而不是直接展示完整 prompt。当前轮不应被理解为“只有一个固定 Context”，而应被理解为“一轮用户请求触发了若干次模型调用，每次调用都有自己的输入上下文、状态快照和输出”。

推荐叙事顺序：

1. 用户问题
2. Agent 执行时间线
3. 本轮模型调用次数与工具调用次数
4. 选中某一次 `Model Call` 后查看该次调用详情
5. 最终回答
6. 原始 prompt / payload / raw think 深层展开

### 7.3 建议的信息架构

建议工作台至少包含以下区域：

- 左侧：运行控制台，展示线程、阶段、模型调用次数、工具调用次数和示例问题
- 主区：当前任务、执行故事线和最终回答
- 右侧：`Model Call Inspector`，先列出本轮所有模型调用，再展示被选中调用的细节
- 深层展开：证据片段、压缩前后对比、原始 prompt/message payload、原始 think 过程

### 7.4 渐进式披露原则

前端不应默认把所有调试信息一次性展开。应采用分层披露：

- 第一层：本轮发生了什么
- 第二层：本轮发生了几次模型调用，以及是否调用工具
- 第三层：点击某一次模型调用后，展示该次调用的 `Input Context`、`State Snapshot` 和 `Model Output`
- 第四层：模型最终收到的原始内容、完整 tool message、完整 memory、完整 raw think

`Model Call Inspector` 的展示规则：

- 每次调用应有明确编号，例如 `Call #1`、`Call #2`
- 列表中展示该调用的目的、阶段、是否触发工具、是否产出最终回答
- 点击某次调用后，`Input Context` 默认只展示参数名；点击参数名后再展开完整内容
- 常见输入参数包括 `system_prompt`、`user_request`、`memory`、`state`、`tool_message`、`evidence`、`recent_dialogue`
- `State Snapshot` 展示该次调用前后的关键状态字段
- `Model Output` 展示工具调用、回答草稿、最终回答或结构化决策摘要
- 原始 think 过程只通过深层按钮打开，不在调用详情默认展开

当前实现要求：

- 前端真实运行模式不能再根据“是否有工具调用”猜测模型调用次数
- 后端必须在每一次真实 LLM 调用外层记录 `ModelCallRecord`
- SSE 应发送 `model_call` 事件，`/api/context/{thread_id}` 也应返回 `model_calls`
- 每条 `ModelCallRecord` 至少包含 `call_id`、`call_index`、`phase`、`purpose`、`input_context`、`state_snapshot`、`output`、可选 `raw_think`
- 前端只消费后端返回的真实 `model_calls`；仅当后端暂未返回时，才允许使用临时 fallback 显示“等待模型调用记录”

### 7.5 不要做成什么

后续前端不要退回到以下方向：

- 通用 AI 聊天壳
- 只做配色美化、不改信息结构
- 纯监控大盘式界面
- 首页直接堆满原始 prompt 和低层技术细节

## 8. 工程实现边界

### 8.1 第一阶段保留项

第一阶段默认保留：

- LangChain / LangGraph 作为执行底座
- FastAPI 后端形态
- React 前端形态
- 当前学术检索工具链
- Redis / Milvus / Embedding 服务

### 8.2 第一阶段新增显式层

即使保留 LangChain/LangGraph，也应在其外侧逐步建立自定义显式层，用于承载：

- `State Reducer`
- `Context Selector`
- `Context Assembler`
- `Trace Recorder`

后续代码组织应尽量让这些职责清晰可见，而不是完全埋在 middleware 和 message 处理里。

### 8.3 不要做的实现方式

后续开发中应避免：

- 把状态继续塞回自然语言摘要里
- 用消息历史替代结构化状态
- 把所有调试信息耦合在单一 SSE 文本流中
- 在没有清晰状态边界的前提下继续堆 UI 功能

## 9. 后续重构执行建议

如果后续 Agent 或开发者继续推进重构，优先顺序建议如下：

1. 明确运行时数据模型：`event / state / context block / assembly record`
2. 重构后端 Context 链路，建立选择和装配过程
3. 重构 SSE 事件协议，使其能承载状态变化和装配信息
4. 重构前端信息架构，改成工作台式布局
5. 再逐步优化视觉、文案、动效和教学细节

不要先做表层美化，再补底层架构。

## 10. 文档维护约定

当本文件内容与实际实现出现偏差时，后续修改应遵循以下原则：

- 优先更新实现，使其向本文件定义的方向收敛
- 若方向变化，先更新 `AGENTS.md`，再开始大规模重构
- 重要架构取舍应写清楚“为什么这样做”，避免只记录结果不记录原则

本文件的职责不是记录所有实现细节，而是作为后续重构的一致性总纲。

## 12. Agentic 工具调用边界

规则路由已从当前实现中移除。后续不要再在后端根据关键词预取 `hybrid_search`，也不要把工具结果提前塞回用户消息中。

当前工具调用原则：

- 是否调用工具由 Agent 根据系统提示词、用户问题和可用工具自行决定
- 后端只负责记录 Agent 实际发起的 `tool_called`、`tool_result_received`、状态更新和 Context 装配
- 前端展示的是 Agent 真实执行轨迹，而不是后端规则预演出来的轨迹

当前不允许的范围：

- 不做关键词触发的工具预取
- 不做复杂多路由器
- 不做模型外部的黑盒路由选择
- 不把规则路由扩展成新的核心框架
- 不为了路由能力牺牲 Agent 调试/演示平台主线

后续优先继续完善：

- State Reducer
- Context Selector
- Context Assembler
- Trace Recorder
- SSE 事件协议
- 前端 Research Workbench 的渐进式披露体验

## 11. 已验证运行注意事项

以下内容来自本仓库实际重构和联调过程，后续继续开发时默认参考，避免重复踩坑。

### 11.1 MiniMax 运行坑

- `MiniMax-M2.7` 在本项目中可正常通过 `OpenAI` 兼容接口调用，`LangChain` 的 `init_chat_model(..., model_provider="openai")` 也可正常工作
- 如果后端进程启动时没有拿到 `OPENAI_API_KEY`，浏览器侧通常表现为：
  - 时间线停在早期阶段
  - 或出现 `agent_error`
  - 后端真实错误通常是 `401 authorized_error`
- 因此，当出现 `MiniMax` 鉴权错误时，优先检查：
  - 当前后端进程是否真的加载了 `.env.local`
  - 而不是先怀疑 `MiniMax` SDK 或 `LangChain` 不兼容

### 11.2 本地服务启动坑

- 在 PowerShell 中停止端口占用进程时，不要使用 `$pid` 作为循环变量名
- `$PID` 是 PowerShell 只读保留变量，误用会导致停止旧进程失败
- 应使用类似 `$procId` 这样的变量名

### 11.3 前端滚动坑

- 当前前端不是单一页面滚动，而是多个独立滚动容器：
  - 侧栏
  - 执行时间线
  - 深层上下文面板
  - 对话区
- 因此浏览器原生“中键自动滚动”不一定能稳定作用于这些内层滚动容器
- 后续如继续保留多层滚动布局，应优先保证这些容器具备显式的中键拖拽滚动能力，而不是假设浏览器默认行为一定可用

### 11.4 事件顺序坑

- 运行时事件时间线不仅要“有事件”，还要保证“发出顺序”和 `sequence` 含义一致
- 对工具链尤其如此：
  - `tool_called` 应先于 `awaiting_tool_result`
  - `tool_result_received` 应先于 `processing_tool_result`
- 否则前端虽然能收到数据，但时间线叙事会错位，演示效果会失真

### 11.5 终端编码坑

- 在 PowerShell 或部分命令行烟测中，中文请求和中文 SSE 内容可能出现乱码或问号
- 这不一定代表模型输出或后端逻辑有问题，可能只是终端编码问题
- 涉及中文链路验证时，优先级建议如下：
  1. 浏览器页面实际行为
  2. UTF-8 配置过的 Python 脚本验证
  3. 最后才是 PowerShell 直接输出

## 13. 当前 Model Call Inspector 落地约定

以下约定覆盖早期关于“参数名 + 前 20 字预览”的旧描述。

当前前端不是框架 demo，而是只消费真实后端 SSE 与 `/api/context/{thread_id}` 返回的 `model_calls`。

右侧 Inspector 的常态结构固定为三块：

- `Model Calls`
- `State Snapshot`
- `Model Output`

交互规则：

- `Model Calls` 列表只显示稳定编号，例如 `call_1`、`call_2`
- 点击某个 `call_1` 后，在右侧展开该次调用的详情
- 再次点击同一个 `call_1` 必须回退到常态，不继续占用右侧空间
- `State Snapshot` 和 `Model Output` 默认保持折叠，通过渐进式披露查看完整内容
- raw think 只能在 `Model Output` 的深层按钮中打开，不进入主对话区或默认展开区

`Input Context` 不再按扁平字段展示。后端记录的 `ModelCallRecord.input_context` 应优先组织成“上下文大块 -> 小块”的结构：

- 大块示例：`Input: Instructions`、`Input: Conversation`、`Input: Tool Evidence`
- 小块示例：`system_prompt`、`user_request`、`recent_dialogue`、`memory`、`tool_message`
- 前端先展示大块，点击大块后再展示小块，点击小块后才展示完整内容

前端主区保持类似 GPT 网页的生成体验：

- 中间主区只承担用户和 Agent 的正常对话
- 模型生成中应自动跟随最新输出滚动到底部
- 不要把 Execution Story、完整 prompt 或低层技术细节塞回主对话区

视觉方向当前定为偏科幻运行数据控制台：

- 深色背景、网格、星点、扫描线、HUD 光环、霓虹边框可以保留
- 动效要服务状态变化和渐进式披露，不要只做无意义装饰
- 若继续改版，应优先提升信息层级和可读性，再增加视觉元素

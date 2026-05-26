# ADR 0002: 以进程内方式交付 `knowledge_expert` 工具，而非通过 MCP stdio

- **状态**: Accepted
- **日期**: 2026-05-10
- **决策者**: research-agent 维护者
- **阶段**: 4.6（RAG 收尾）
- **取代**: Phase-2 中将 `knowledge_server` 包装为与 `fin_data_server`、
  `pdf_report_server` 和 `code_server` 相同的 MCP stdio 子进程的计划。

## 背景

项目的工具交付契约是 **Model Context Protocol (MCP) over stdio**：
每个工具族（金融数据、PDF 报告、代码执行、新闻……）都实现为一个
`fastmcp` 服务器，在 FastAPI 启动时由
`langchain_mcp_adapters.MultiServerMCPClient` 作为子进程启动。
Supervisor 从不导入工具实现；它只看到 JSON-RPC 信封。

这种模式具有我们想要的三个属性：

1. **严格的工具/Agent 隔离。** 工具崩溃不会拖垮 Agent。
2. **语言无关的工具。** Node.js MCP 服务器与 Python 服务器同样合法。
3. **工具实现热替换。** 用新版本替换 `fin_data_server.py`，Agent
   下次启动时即可感知，无需代码变更。

因此 Phase-2 设计规定：也将 `knowledge_server` 实现为 MCP 服务器。
这编写起来很简单 — 文件存在，用 `@mcp.tool()` 装饰 — 但实际上
`knowledge_server` 子进程在 Windows 上间歇性卡死。诊断：

- 卡死发生在业务逻辑之后：`_ingest` 函数运行完成，返回一个 `dict`，
  `fastmcp` 开始将响应序列化到 `stdout`。父进程从未看到响应，
  在 90 秒后超时。
- 根本原因是 (a) 与 MCP 传输在同一进程中的重量级 ML 导入
  （`sentence-transformers`、`langchain_text_splitters`，最初还有
  `chromadb` — 参见 [ADR 0001](0001-faiss-over-chroma.md)）和
  (b) Windows 的 stdio 管道行为与来自原生 C 扩展和 Python 的
  混合缓冲/非缓冲写入的结合。
- 我们尝试了一个底层修复：一个 `_stdio_firewall.py` 模块在服务器
  导入时运行，`dup` 原始 `stdout` fd，将 fd 1 重定向到 `NUL`
  以静默 C 级别写入，并将 Python 的 `sys.stdout` 通过保存的 fd
  重路由回来。这对 ChromaDB 的守护线程杂音有效，但**未能**防止
  FAISS 下 `_ingest` 后的卡死 — 证明堆栈中不止一层在贡献。

相同的 `knowledge_server.py` 代码，以**进程内**方式调用（无 MCP
子进程、完全无 stdio），清洁地执行 `_ingest` + `_search` + `_list`
+ `_delete`，远在原始超时之内。我们通过 `_diag_ingest_inproc.py`
（现已删除）确认了这一点。

## 决策

将知识库操作以**进程内 LangChain `StructuredTool` 实例**的形式暴露给
Agent，而非 MCP stdio 工具。

机制：

- `src/research_agent/mcp_servers/knowledge_server.py` 保留其
  `@mcp.tool()` 装饰器，**名义上**仍是一个 MCP 服务器 — FastMCP
  入口点被保留，因此当需要时（如从非 Python Agent 调用，或用于
  协议一致性测试）仍可作为子进程运行。
- `src/research_agent/tools/knowledge_tools.py` 是生产模式适配器：
  它直接导入 `_ingest`、`_search`、`_list_collections`、
  `_delete_collection`，并将每个包装在 `StructuredTool.from_function`
  中。这些才是 `knowledge_expert` 在运行时实际消费的。
- `src/research_agent/mcp_servers/client_factory.py` 暴露
  `load_knowledge_tools_inproc()` — 与其他 `load_*_server_tools()`
  函数形状对称 — 因此 FastAPI lifespan 在调用点对待知识库工具与
  MCP 加载的工具完全一致。仅实现不同。

## 考虑过的替代方案

1. **通过 MCP SSE 而非 stdio 运行 `knowledge_server`。**
   FastMCP 支持 SSE 传输，可以绕开 stdio 管道问题。否决原因：
   - 它在部署中增加了一个 TCP 端口（或 Unix 域套接字，Windows
     上不存在）。
   - 它将故障模式从"子进程卡死"转变为"子进程崩溃；SSE 客户端
     在心跳超时前一直阻塞 Agent"。
   - 在撰写时，`langchain_mcp_adapters` 中 SSE 传输不如 stdio 成熟。
2. **用 `PYTHONUNBUFFERED=1` 和
   `sys.stdout.reconfigure(line_buffering=True)` 启动 MCP 子进程。**
   已尝试；在重量级 ML 导入下 Windows 上减少但未消除卡死。
3. **将重量级 ML 导入移到惰性路径。** `_ingest` 已经在函数内部
   惰性导入 `sentence-transformers` 和 `langchain_text_splitters`。
   卡死发生在 `_ingest` 返回**之后**，在写入响应时，因此惰性导入
   无济于事。
4. **接受超时并在 supervisor 级别添加重试逻辑。** 否决原因：重试
   浪费嵌入工作（sentence-transformers 调用开销不低）并膨胀延迟。
   同时掩盖了真正的问题。
5. **将所有专家移到进程内。** 对称性很诱人，但会丢失我们从 MCP
   对其他专家所期望的属性：`fin_data_server` 调用 akshare 时可能
   因上游 API 而合法崩溃；隔离是有价值的。我们仅为这一个导入链
   与 stdio 不兼容的专家承担"两种传输"成本。

## 后果

### 正面

- **可靠性。** 知识库操作再次具有确定性。不再有 Windows 上的 90 秒
  超时；不再有"在 Linux 上正常，在同事笔记本上卡死"。
- **知识库调用冷启动更快。** 无逐调用的子进程启动；工具即方法调用。
  本地基准：服务器启动后首次 `knowledge_search` 从约 3.2 秒降至
  约 0.4 秒。
- **急切导入预热仍有效。** `knowledge_server.py` 顶部的模块级导入
  （FAISS、文本分割器）将昂贵的初始化前置到 FastAPI 启动时，使首次
  `knowledge_ingest_pdf` 调用不必承担该开销。最初引入是为了避免
  MCP-stdio 死锁，急切导入作为启动预热机制保持其价值。
- **MCP 协议表面保留。** `@mcp.tool()` 装饰器完好无损，因此服务器
  仍可由非 Python 客户端或在协议一致性测试中作为 stdio 子进程运行。
  默认的 Python Agent 路径只是选择了进程内适配器。

### 负面

- **知识库工具失去了崩溃隔离。** `_ingest` 中未处理的异常（如损坏的
  PDF 导致 pypdf 在调用任务内抛出异常）现在落在 FastAPI worker 内部。
  我们通过用防御性 try/except 包装每个进程内工具函数来缓解 — 将异常
  转换为结构化错误返回 — 与 MCP 错误信封的契约相同。
- **需维护两种交付模式。** `knowledge_server.py` 文件现在服务两个
  消费者：MCP 运行时（用于协议一致性）和进程内适配器（用于 Agent）。
  进程内路径需要的任何东西（惰性/急切导入、错误包装）都必须与子进程
  路径保持兼容。我们在模块文档字符串中明确记录了这一点。
- **解释叙事中的对称性打破。** 在讲解架构时，"所有六个专家都由 MCP
  支持"是一个清晰的一句话。现实是"五个 MCP、一个进程内；原因如下，
  以及我们如何保持协议表面完整"。这是一个更差的表述，但更准确。

### 中性

- Supervisor 和 Agent 代码的其余部分不感知交付模式。它们以任何方式
  消费 `BaseTool` 实例。
- `knowledge_expert` 仍在构建时验证正确的工具族已加载（它拒绝空的
  工具列表）— 契约与其他专家完全相同。

## 状态

在提交 `<git rev-parse HEAD>`（Phase 4.6）中实现。全部
167 + 21 + 2 个单元测试（回归 + 反思 + 新包装器测试）在
Windows + Python 3.13.13 上为绿色。知识库端到端流程
（灌入 → 列举 → 搜索 → 删除）通过进程内测试固件验证。

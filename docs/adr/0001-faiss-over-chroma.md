# ADR 0001: 使用 FAISS（文件存储）代替 ChromaDB 作为知识库

- **状态**: Accepted
- **日期**: 2026-05-10
- **决策者**: research-agent 维护者
- **阶段**: 4.6（RAG 收尾）

## 背景

`research_agent` 在 `knowledge_expert` 专家后面提供了一个按用户隔离的
PDF 知识库。用户上传 PDF，服务器执行 解析 → 分块 → 嵌入 → 存储，搜索时
Agent 执行混合检索（向量 + BM25）并重排序。向量存储位于每个研究会话的热路径上，
因此其运营特性主导了开发者体验。

最初的 Phase-0 计划选择了 **ChromaDB** 担任此角色，因为：

- 持久化磁盘存储，无需单独进程 — 符合我们首先期望的"单用户、
  单笔记本"部署模式。
- 开箱即用的基于 HNSW 的近似最近邻搜索。
- 熟悉的 API 接口（`langchain-chroma`），与 LangChain 生态系统的
  其余部分映射清晰。

Phase-4 期间发生了三件事，迫使我们重新审视这一选择：

1. **Windows 上的 stdio 管道损坏。** `knowledge_server` MCP 子进程
   通过 `stdin`/`stdout` 上的 JSON-RPC 运行。导入 `chromadb` 会拉入一个
   包含 `posthog`（遥测）、`protobuf` 和 `onnxruntime` 的链。在
   Python 3.13 / Windows 上，我们观察到子进程的 `stdout` 管道变成部分
   缓冲模式，导致 JSON-RPC 响应延迟 — `_ingest` 完成执行，将响应写入
   `sys.stdout`，而父进程在 90 秒后超时但什么也没收到。我们通过一个
   进程内诊断（调用完全相同的函数并在数秒内完成）确认这是 stdio 层的问题
   （而非 `_ingest` 运行时问题）。一个定制的 `_stdio_firewall.py`
   将 fd 1 重定向到 `NUL` 并将 `sys.stdout` 重路由到保存的 fd，
   减少但未完全消除症状。
2. **禁用遥测不够充分。** 设置 `ANONYMIZED_TELEMETRY=False` 和
   `CHROMA_TELEMETRY=False` 可移除遥测**守护线程**，但 `chromadb`
   本身会启动后台 worker（段管理器、嵌入队列），其日志 — 即使被静默 —
   通过 Python 的导入时 `logging` 配置，仍然在快速 `_ingest` 工作负载下
   影响 stdio 管道。
3. **构建/安装体积。** `chromadb` 拉入的传递依赖集解压后约 120 MB
   （`onnxruntime`、`chroma-hnswlib`、`tokenizers`、`tqdm`、`mmh3`、
   `bcrypt`、`pulsar-client`……）。对于一个已经包含 `sentence-transformers`、
   `langchain-*` 和 MCP 运行时的项目，额外的重量使 Docker 镜像重建明显变慢，
   且在全新笔记本上 `uv sync` 冷路径超过 5 分钟。

## 决策

用 **FAISS**（通过 `faiss-cpu` 和
`langchain_community.vectorstores.FAISS`）替换 `chromadb` +
`langchain-chroma` 作为知识库的磁盘向量存储。

具体来说：

- 每个 collection 是磁盘上的一个目录
  （`./data/knowledge_db/<collection_name>/`），包含
  `index.faiss` + `index.pkl`。加载使用 `FAISS.load_local(...)`，
  保存使用 `vs.save_local(...)`。无守护进程、无遥测、无后台线程、
  无需绑定端口。
- 现有的双路检索逻辑（向量 + BM25 + 重排序器）保持不变 —
  仅更换向量后端。
- Collection 删除即 `shutil.rmtree` 目录；collection 列举即
  `os.listdir` 根目录，以 FAISS 索引文件存在为门控。两个操作
  都是 O(1) 系统调用，而非到 ChromaDB 客户端的往返。

## 考虑过的替代方案

1. **继续使用 ChromaDB，正确修复 Windows stdio 问题。**
   需要向 chromadb 上游提交补丁修改其 logging / 初始化顺序，或 fork 它
   以移除段管理器的嘈杂副作用。两者都是无产品价值的带外投入；stdio 问题
   是 Windows 特有的开发者体验问题，在 Linux/macOS 上根本不存在，但项目的
   MVP 必须在 Windows 上运行。
2. **将 ChromaDB 作为独立 HTTP 服务器运行。** 官方 `chromadb run`
   模式可以完全绕开 stdio 问题（Agent 和向量存储之间不共享 `stdout`）。
   否决原因：它使单用户开发环境回到多进程部署，最低限度地增加了
   `docker-compose` 复杂度，但在 README 中增加了一个端口，以及项目
   不愿拥有的权限管理。
3. **Pinecone / Weaviate / Qdrant 云。** 生产级但需要 API 密钥、
   网络和逐调用延迟。对于为"克隆即运行"优化的项目来说是错误选择。
4. **在现有 Postgres 容器中使用 pgvector。** 有吸引力，因为项目已经
   为 checkpointing 运行 Postgres，但会迫使每个用户启动 Postgres 才能
   运行知识库 — 这与我们为 Phase-4.6 冒烟测试所期望的"knowledge_expert
   可独立工作"属性相矛盾。我们保留 pgvector 用于未来 Phase-6 的
   "生产级"模式。

## 后果

### 正面

- **零安装摩擦。** `pip install faiss-cpu` 是单个约 15 MB 的 wheel；
  无 `onnxruntime`、无遥测、无守护进程。Windows 上冷 `uv sync`
  时间从 > 5 分钟降至约 90 秒。
- **稳定的 stdio 行为。** `knowledge_server` 的导入链不再触及后台线程
  或 fd 级别的 logging。剩余的卡死（[ADR 0002](0002-knowledge-server-inprocess.md)）
  与 ChromaDB 无关，且在 FAISS 迁移后依然存在 — 这本身证明 ChromaDB
  至少是*其中一个*原因。
- **更简单的运营模型。** 备份 = `tar` collection 目录。迁移 = 复制目录。
  删除 = `rm -rf`。不存在"vacuum" / "compact" / "一致性检查"维护任务。

### 负面

- **无增量写入持久性。** FAISS 通过将整个索引序列化回磁盘来持久化。
  我们在每次成功的 `_ingest` 调用后同步执行此操作，对于数百 MB 的向量
  没问题，但无法扩展到百万级分块的流式灌入。当前产品目标是"一个分析师的
  PDF 库，≤ 几百份文档"，因此可以接受；如果我们跨越 10⁵ 分块阈值，
  未来的 ADR 将用 pgvector 或 LanceDB 替换 FAISS。
- **无多进程写入安全性。** 两个进程同时写入同一 collection 目录会在
  `save_local` 上竞争。我们通过将所有知识库写入操作集中到 FastAPI 进程内
  单个 `knowledge_expert` Agent 来缓解（自 [ADR 0002](0002-knowledge-server-inprocess.md)
  起，该 Agent 本来就在进程内运行）。第二个写入者需要文件锁 — 多租户
  之前不在范围内。
- **HNSW 可调参数较不方便。** ChromaDB 将索引参数暴露为单个配置字典；
  FAISS 在索引构建时暴露它们。我们对 ≤ 10⁴ 分块默认使用 FAISS 的
  `IndexFlatL2`，因为精确搜索在该规模下速度很快，且延迟由嵌入而非 ANN
  主导。当（如果）我们切换到 `IndexHNSWFlat` 时，会在更新的 ADR 中
  记录权衡。

### 中性

- 嵌入模型（`BAAI/bge-small-zh-v1.5`）和重排序器（`BAAI/bge-reranker-base`）
  不受影响 — 它们从未与向量存储选择绑定。
- BM25 辅助索引（从向量存储持有的同一文档库构建）保持不变；唯一的编辑
  是从 FAISS 的 docstore 而非 ChromaDB 的 collection 加载文档。

## 状态

在提交 `<git rev-parse HEAD>`（Phase 4.6）中实现。Linux 和 Windows
冒烟测试通过；灌入 + 搜索 + 删除 + 列举 循环在
`tests/unit/test_mcp_echo_server.py` 风格的测试框架中为绿色（知识库测试
在进程内运行，参见 ADR 0002）。

# ADR 0005: FAISS → pgvector 迁移路径

- **状态**: Accepted
- **决策者**: research-agent 维护者

## 背景

Phase 4 选择 FAISS 作为向量存储（见 ADR-0001），原因是 ChromaDB 在 Windows MCP stdio 上不稳定。
FAISS 稳定可靠，但有固有局限：

1. **文件级锁** — 不支持并发写入；多实例部署时只能各自维护独立副本。
2. **无 SQL 查询** — 无法按 metadata 做复合过滤（如 `source = 'annual_2024.pdf' AND page > 10`）。
3. **额外进程** — docker-compose 中已有 `pgvector/pgvector:pg16` 容器，运行着带 pgvector 扩展的 Postgres。向量和关系数据可统一管理。

## 决策

引入**向量存储抽象层** (`rag/vector_backend.py`)，实现双后端：

- `FaissVectorBackend` — 委托现有 knowledge_server FAISS 逻辑，零改动
- `PgvectorBackend` — 通过 `rag/pgvector_store.py`（psycopg + pgvector）连接已有 Postgres

通过 `KNOWLEDGE_VECTOR_BACKEND` 环境变量切换：

```
# .env
KNOWLEDGE_VECTOR_BACKEND=faiss     # 默认，开发环境
KNOWLEDGE_VECTOR_BACKEND=pgvector  # 生产环境
```

### 迁移策略：渐进式

1. **Phase 1（当前）**：抽象层就绪，默认仍用 FAISS。knowledge_server 内部逻辑不变。
2. **Phase 2（未来）**：knowledge_server 的 `ingest_pdf` 和 `search` 改为调用 `get_vector_backend()` 而非直接操作 FAISS。
3. **Phase 3（未来）**：提供 `scripts/migrate_faiss_to_pgvector.py` 一键迁移脚本。

### 为什么不一步到位

- 现有 85+ 行的 knowledge_server FAISS 逻辑已经过充分测试
- BM25 索引依赖 FAISS docstore 的内存结构，pgvector 需要单独的 BM25 索引策略
- 对演示而言，展示"设计了抽象层 + 渐进迁移路径"比"全部重写"更体现工程素养

## 备选方案

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| 全量迁移到 pgvector | 架构统一 | 破坏现有测试；BM25 需重新实现 | 风险过高 |
| 保持 FAISS 不变 | 零改动 | 无法多实例；面试时缺乏迁移思考 | 不展示演进能力 |
| **抽象层 + 渐进迁移**（选定） | 向后兼容；展示设计能力 | 初期有两套实现 | 最佳平衡 |

## 后果

### 正面
- 可展示"为什么选 FAISS → 什么时候该迁 → 怎么迁"的完整思考链
- 生产部署时切一个环境变量即可使用 pgvector
- 不破坏任何现有功能和测试

### 负面
- pgvector 后端尚未被 knowledge_server 主路径使用（Phase 2 工作）
- pgvector 后端使用项目已有的 `psycopg` + `pgvector` 依赖，无需额外安装

### 风险
- pgvector 后端的 `similarity_search_with_score` 返回的分数语义可能与 FAISS 不同（pgvector 返回距离而非相似度）— 需在 Phase 2 接入时统一归一化

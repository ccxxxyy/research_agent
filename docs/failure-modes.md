# 故障模式分析

本文档分析 Research Agent 的关键故障模式、影响范围、检测手段与缓解策略。

## 一、故障模式矩阵

| 组件 | 故障模式 | 影响 | 检测 | 缓解 | 恢复 |
|------|---------|------|------|------|------|
| **LLM 提供商 (HEAVY)** | API 超时 / 5xx | 研究请求失败或延迟飙升 | 熔断器连续失败计数；`/metrics` 中 `llm_calls_total` 异常 | 熔断器 → fallback 到 MEDIUM 模型 | 30s 后半开状态自动探测恢复 |
| **LLM 提供商 (全部)** | API Key 过期 / 额度耗尽 | 所有 LLM 调用失败 | 401/403 状态码；日志中连续认证错误 | 无自动缓解（需人工更换 Key） | 更新 `.env` 中的 API Key 并重启 |
| **MCP 子进程 (fin_data)** | `uv run` 启动失败 / 超时 | data_expert 不可用，其他专家正常 | 启动日志 `Tool discovery failed for fin_data_server` | Supervisor prompt 自动裁剪该专家 | 重启应用或修复依赖 |
| **MCP 子进程 (全部)** | 所有子进程启动失败 | 研究端点返回 503 | `/health` 返回 specialist 列表为空 | 503 明确告知客户端；minimal supervisor 仍可用 | 检查网络/依赖后重启 |
| **PostgreSQL** | 连接被拒 / 超时 | 检查点和长期记忆不可持久化 | 启动日志 `PostgreSQL unreachable, falling back to SQLite` | 自动降级到 SQLite | PostgreSQL 恢复后重启应用切回 |
| **PostgreSQL + SQLite** | 磁盘满 / 文件锁冲突 | 检查点写入失败 | 日志中 `IOError` / `OperationalError` | 降级到 InMemoryStore（非持久化） | 清理磁盘空间 |
| **Redis** | 连接失败 / 超时 | 限流和 Token 配额回退到内存 | 日志 `Redis unavailable, using in-memory fallback` | 透明降级到进程内 dict（单实例有效） | Redis 恢复后自动重连（下次请求） |
| **RAG 向量存储** | FAISS 索引文件损坏 | knowledge_expert 检索返回空结果 | `quality: "low"` 标签；Corrective RAG 多次重写无改善 | Corrective RAG 最多重试 3 次，最终返回 "未找到相关内容" | 重新 ingest PDF 重建索引 |
| **RAG 重排序模型** | 模型权重未下载 / 加载 OOM | CrossEncoder 跳过，退回 RRF 粗排序 | 日志 `Reranker disabled or failed to load` | 降级到纯 RRF 融合排序（精度下降但可用） | 手动下载模型到 cache 目录 |
| **网络 (外部 API)** | akshare / 东财接口不可达 | 对应专家返回 error dict | 专家输出含 `"error"` 键 | Supervisor 如实说明错误，不捏造数据 | 等待外部 API 恢复 |
| **内存** | 进程 OOM | 整个 API 被系统 kill | Docker healthcheck 失败；监控告警 | Docker restart policy `unless-stopped` | 自动重启；排查内存泄漏 |
| **并发** | 超过限流阈值 | 部分请求被 429 拒绝 | `/metrics` 中 429 计数上升 | 客户端根据 `Retry-After` 头重试 | 调整 `RATE_LIMIT_RPM` 配置 |
| **Token 配额** | 单用户日配额耗尽 | 该用户后续请求被 429 拒绝 | Token 配额管理器返回 `(False, 0)` | 返回 429 + 剩余配额信息 | 24h 窗口滑过后自动恢复 |

## 二、降级策略详解

### 2.1 存储三级降级

```
PostgreSQL (生产级持久化)
    │ 连接失败
    ▼
SQLite (本地文件持久化)
    │ 文件操作失败
    ▼
InMemoryStore (非持久化，重启丢失)
```

决策在 lifespan 启动时一次性完成，运行时不再切换——避免中途切换导致的状态不一致。

### 2.2 LLM 熔断降级

```
HEAVY (deepseek-v4-pro)
    │ 连续 3 次失败
    ▼ [CircuitBreaker OPEN]
MEDIUM (qwen3.6-plus)    ← with_fallbacks 自动切换
    │ 连续 3 次失败
    ▼ [CircuitBreaker OPEN]
抛出异常 → 路由返回错误（不静默返回低质量结果）
```

### 2.3 MCP 工具发现降级

```
全部 6 个 specialist 工具加载成功 → 完整能力
    │ 部分失败
    ▼
仅注册成功的 specialist → Supervisor prompt 自动裁剪 → 部分能力
    │ 全部失败
    ▼
研究端点返回 503 → Minimal supervisor (3 个 toy 工具) 仍可用
```

## 三、可观测性信号

| 故障类别 | 监控指标 | 日志关键词 | 告警建议 |
|---------|---------|-----------|---------|
| LLM 失败 | `research_agent_llm_calls_total` 异常下降 | `Circuit breaker OPEN` | 熔断器 OPEN 持续 > 2 分钟 |
| API 错误率 | `research_agent_http_requests_total{status="5xx"}` | `500`, `503` | 5xx 比例 > 5% |
| 延迟退化 | `research_agent_http_request_duration_seconds` P95 | `Request timed out` | P95 > 30s |
| 限流触发 | `429` 状态码计数 | `Rate limit exceeded` | 429 比例 > 10% |
| 存储降级 | — | `falling back to SQLite`, `InMemoryStore fallback` | 启动时非 PostgreSQL |
| MCP 失败 | `research_agent_specialists_available` 下降 | `Tool discovery failed` | specialist 数量 < 3 |

## 四、灾难恢复

### 场景 1：LLM 提供商全面宕机

1. 熔断器自动 OPEN，所有请求走 fallback 链直到耗尽
2. API 返回 500 错误，前端展示 "服务暂时不可用"
3. **恢复**：提供商恢复后，熔断器 30s 内自动 HALF_OPEN → CLOSED

### 场景 2：PostgreSQL 不可达

1. 启动时自动降级到 SQLite（或 InMemory）
2. 检查点和长期记忆写入本地文件
3. **恢复**：修复 PostgreSQL 后重启应用，新数据写入 PG；SQLite 期间的数据需手动迁移

### 场景 3：应用进程崩溃

1. Docker `restart: unless-stopped` 自动重启
2. lifespan 重新初始化所有资源
3. PostgreSQL 检查点保证跨重启的对话状态恢复
4. **恢复**：自动（SQLite/PG 检查点持久化保证状态不丢失）

### 场景 4：磁盘空间耗尽

1. SQLite 写入失败，降级到 InMemoryStore
2. 日志写入失败（loguru 的 rotation 机制会尝试清理旧日志）
3. **恢复**：清理日志/临时文件 → 重启

# ADR 0004: 多层安全防御体系（Guardrails）

- **状态**: Accepted
- **日期**: 2026-05
- **决策者**: research-agent 维护者

## 背景

本系统是面向 A 股市场的金融研究 AI 助手。金融场景对安全性有特殊要求：

1. **Prompt 注入风险**：用户输入或第三方数据源中可能包含恶意指令，诱导 LLM 泄露系统提示词、生成虚假数据、或绕过路由规则。
2. **金融合规风险**：LLM 可能生成不当投资建议（"建议买入"、"保证收益"等），在中国证券法框架下可能构成违规。
3. **资源滥用风险**：单一用户可能通过大量请求耗尽共享 LLM Token 预算。
4. **信息泄漏风险**：LLM 输出可能包含系统提示词、内部路径、API 凭据等敏感信息。

现有措施（P0 阶段之前）仅有基本的 Auth 中间件和 IP 限流，缺乏针对 LLM 特有风险的防御。

## 决策

实施四层安全防御体系：

### 第一层：输入安全（PromptGuard — 输入规则引擎）

基于正则的快速检测（微秒级），覆盖中英文双语 15+ 种注入模式：
- 指令覆盖（忽略之前指令 / ignore previous instructions）
- 角色劫持（你现在是 / act as）
- 系统提示词提取（输出你的系统提示 / print system prompt）
- 越狱模板（DAN / 开发者模式 / 无限制模式）
- 间接注入标记（IMPORTANT NEW INSTRUCTION / 重要新指令）
- 编码绕过（base64 decode / rot13）

威胁分级：`BLOCKED` → 400 拦截，`SUSPICIOUS` → 放行但记录日志。

### 第二层：输出安全（PromptGuard — 输出规则 + 金融合规）

- 系统提示词逐字泄漏检测
- API Key / 凭据泄漏检测
- 内部路径泄漏检测
- 不当投资建议检测（"建议买入"、"保证收益"等）
- 金融免责声明自动附加到所有研究类输出

### 第三层：流量控制

- **IP 限流**（RateLimitMiddleware）：滑动窗口 RPM，Redis 分布式 + 内存兜底
- **Per-user Token 配额**（TokenQuotaManager）：24h 窗口内 Token 总量上限，防止单用户耗尽预算

### 第四层：基础设施安全

- Bearer token 认证（AuthMiddleware）
- 请求超时（RequestTimeoutMiddleware）— SSE 长连接豁免
- 请求 ID 追踪（RequestIdMiddleware）— 全链路日志关联
- CORS 策略

## 备选方案

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| **LLM-based 二次验证** | 精度更高，能捕捉语义级注入 | 每次请求增加一次 LLM 调用，延迟 +2-5s，成本翻倍 | 作为可选增强层保留（当前不启用） |
| **第三方 Guardrails 服务** (NeMo, Guardrails AI) | 成熟方案，社区维护 | 引入外部依赖，增加部署复杂度；国内网络延迟不可控 | 不采用 |
| **纯 LLM 提示词约束** | 零代码改动 | 可被 jailbreak 绕过；无法防御编码绕过等技术手段 | 不充分 |

## 后果

### 正面

- 覆盖 OWASP Top 10 for Agentic AI 中的 LLM01（Prompt Injection）和 LLM02（Insecure Output Handling）
- 金融免责声明降低合规风险
- Per-user 配额防止资源滥用
- 纯规则引擎，零额外 LLM 成本，微秒级延迟

### 负面

- 正则规则可能误报（已通过精准模式和 SUSPICIOUS 分级缓解）
- 金融免责声明可能影响用户体验（但合规优先）
- Token 配额可能过于保守，需根据实际使用调整默认值

### 风险

- 语义级注入（不含关键词的指令覆盖）无法被纯规则引擎捕获——需要后续叠加 LLM-based 检测层
- 中文 NLP 的多义性可能导致金融合规规则误判——需持续维护规则库

# 架构决策记录

本目录存放项目的 ADR（架构决策记录）— 那些承重级的架构选择，
在做出决策的当下记录下来，包含我们否决的替代方案及**原因**。
目标有二：

1. **给未来的维护者（包括未来的你）：** 回答"我们为什么不直接
   做 X？"的问题，而不必从零开始重新辩论。
2. **用于面试 / 代码审查：** 证明代码库中那些不显而易见的决策
   是经过深思熟虑的权衡，而非偶然之举。

## 格式

我们使用 [Michael Nygard ADR 模板](https://github.com/joelparkerhenderson/architecture-decision-record/tree/main/locales/en/templates/decision-record-template-by-michael-nygard)
的精神，略有改编：

- **Status（状态）**: `Proposed` → `Accepted` → `Superseded by …` /
  `Deprecated`。一旦写成，ADR 是不可变的；后续决策以新文件记录。
- **Context（背景）**: 我们正在解决的问题。
- **Decision（决策）**: 我们选择了什么。
- **Alternatives considered（考虑过的替代方案）**: 我们否决了什么，及理由。
- **Consequences（后果）**: 正面、负面和中性的影响。

## 索引

| #    | 标题                                                                    | 状态     | 阶段  |
|------|-------------------------------------------------------------------------|----------|-------|
| 0001 | [使用 FAISS（文件存储）代替 ChromaDB 作为知识库](0001-faiss-over-chroma.md) | Accepted | 4.6   |
| 0002 | [以进程内方式交付 `knowledge_expert` 工具，而非通过 MCP stdio](0002-knowledge-server-inprocess.md) | Accepted | 4.6   |
| 0003 | [添加 Writer / Reasoner 反思循环作为 supervisor 后置子图](0003-reflection-loop.md) | Accepted | 5.2   |
| 0004 | [多层安全防御体系（Guardrails）](0004-guardrails-security-layers.md) | Accepted | — |
| 0005 | [pgvector 迁移路径](0005-pgvector-migration-path.md) | Accepted | — |
| 0006 | [A 股 / 美股平行隔离与市场判定契约（P0）](0006-us-market-parallel-isolation.md) | Accepted | US-P0 |

## 何时编写新的 ADR

当以下**任何一项**成立时编写：

- 我们选择了非默认方案（如选择了库 A 而非库 B；以进程内方式
  运行而非进程外）。
- 该决策涉及多个模块 / 包。
- 合理的读者事后会问"为什么是这样的？"
- 该决策在运营中有可感知的后果（额外延迟、额外依赖、额外故障模式）。

以下情况**不要**编写：常规重构、错误修复、或不具有有意义替代
方案的实现细节。

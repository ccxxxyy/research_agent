"""研究 supervisor 的量化评估套件。

组件：
- ``evaluators`` — 7 个评估器（routing_accuracy / reply_quality / keyword_coverage / memory_persistence / tool_selection_precision / market_routing_accuracy / market_isolation）
- ``eval_supervisor`` — LangSmith 在线评估入口
- ``run_local`` — 离线评估 runner，输出 JSON 报告到 eval_results/
- ``compare`` — 两次评估报告的 regression 对比工具
- ``datasets/supervisor_routing.json`` — A 股路由样本（110）
- ``datasets/us_market_routing.json`` — 美股路由 / 隔离样本
- ``datasets/mixed_market_routing.json`` — 跨市场 MIXED 编排样本

运行方式::

    # LangSmith 在线评估
    python -m evals.eval_supervisor

    # 离线本地评估（默认合并 CN + US）
    python -m evals.run_local
    python -m evals.run_local --cn-only
    python -m evals.run_local --dataset evals/datasets/us_market_routing.json

    # 对比 regression
    python -m evals.compare eval_results/baseline.json eval_results/current.json

    # 评估器单元测试
    pytest evals/test_evaluators.py evals/test_us_eval_dataset.py -q
"""

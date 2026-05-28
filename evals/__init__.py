"""研究 supervisor 的量化评估套件。

组件：
- ``evaluators`` — 5 个评估器（routing_accuracy / reply_quality / keyword_coverage / memory_persistence / tool_selection_precision）
- ``eval_supervisor`` — LangSmith 在线评估入口
- ``run_local`` — 离线评估 runner，输出 JSON 报告到 eval_results/
- ``compare`` — 两次评估报告的 regression 对比工具
- ``datasets/supervisor_routing.json`` — 100 条标注样本

运行方式::

    # LangSmith 在线评估
    python -m evals.eval_supervisor

    # 离线本地评估
    python -m evals.run_local

    # 对比 regression
    python -m evals.compare eval_results/baseline.json eval_results/current.json

    # 评估器单元测试
    pytest evals/test_evaluators.py -q
"""

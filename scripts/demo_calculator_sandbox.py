"""演示保护 ``calculate`` 免受代码注入的双重安全门。

运行::

    uv run python scripts/demo_calculator_sandbox.py

本脚本 不是 单元测试 — 它是一个带叙述的演示，适合对照源码阅读。它模拟攻击者的思维模型：

    攻击者视角:  "calculate() 使用 eval。我肯定能逃逸。"
    防御者视角:  "两道门。给我看看哪个能穿过去。"

对于每次尝试打印：
  1. 原始载荷
  2. 编译期的 ``co_names`` 元组（安全门 1 的依据）
  3. ``calculate`` 的返回值（防御者的裁决）

最后还运行两个"假如"消融实验：如果只有安全门 1？只有安全门 2？— 使分层防御的论点变得具体可感。
"""

from __future__ import annotations

from research_agent.tools.native import calculate

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _inspect_co_names(expression: str) -> tuple[str, ...]:
    """返回 Python 编译器在运行时将要解析的名称。

    这正是 ``calculate`` 中安全门 1 所检查的内容。通过对每个探针打印此元组，可以看到载荷为何被拒绝。
    """
    try:
        code = compile(expression, "<demo>", "eval")
        return tuple(code.co_names)
    except SyntaxError as e:
        return (f"<SyntaxError: {e.msg}>",)


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _run(label: str, expression: str) -> None:
    names = _inspect_co_names(expression)
    result = calculate.invoke({"expression": expression})
    verdict = "已拦截" if result.startswith("Error:") else "已放行"
    print(f"\n  [{verdict}] {label}")
    print(f"    载荷      : {expression!r}")
    print(f"    co_names  : {names}")
    print(f"    返回值    : {result}")


# ---------------------------------------------------------------------------
# 第 1 部分 — 正常路径（基线：合法数学运算应正常工作）
# ---------------------------------------------------------------------------


def demo_happy_path() -> None:
    _banner("第 1 部分 — 合法数学运算（应全部放行）")

    _run("基本算术", "2 + 3 * 4")
    _run("白名单内置函数", "round(3.14159, 2)")
    _run("白名单 math 模块", "sqrt(16) + pi")


# ---------------------------------------------------------------------------
# 第 2 部分 — 攻击目录（每个都必须被拦截）
# ---------------------------------------------------------------------------


def demo_attacks() -> None:
    _banner("第 2 部分 — 攻击目录（应全部拦截）")

    _run(
        "经典 RCE：通过 __import__",
        "__import__('os').system('echo pwned')",
    )
    _run(
        "文件系统读取：通过 open",
        "open('/etc/passwd').read()",
    )
    _run(
        "环境变量泄露：通过 builtins 链",
        "__builtins__['__import__']('os').environ",
    )
    _run(
        "子类遍历（经典沙箱逃逸）",
        "(1).__class__.__bases__[0].__subclasses__()",
    )
    _run(
        "动态 exec",
        "exec('import os; os.system(\"id\")')",
    )
    _run(
        "嵌套 eval",
        'eval("1+1")',
    )
    _run(
        "globals 检查",
        "globals()",
    )
    _run(
        "从字面量开始的属性遍历",
        "().__class__.__mro__",
    )


# ---------------------------------------------------------------------------
# 第 3 部分 — 语法无效的载荷（在任一安全门之前就被拒绝）
# ---------------------------------------------------------------------------


def demo_garbage() -> None:
    _banner("第 3 部分 — 畸形载荷（SyntaxError 处理）")

    _run("重复运算符", "2 ++ ** 3")
    _run("未闭合括号", "((1+2)")


# ---------------------------------------------------------------------------
# 第 4 部分 — 消融实验：为什么需要两道安全门
# ---------------------------------------------------------------------------


def _unsafe_gate1_only(expression: str) -> str:
    """假设的 ``calculate`` 变体，**仅有** 安全门 1（名称白名单）。

    注意：``__builtins__`` 未被清空，因此任何通过白名单的名称都会解析到真实的内置函数。
    """
    allowed = {"abs": abs, "round": round, "min": min, "max": max}
    code = compile(expression, "<gate1-only>", "eval")
    for name in code.co_names:
        if name not in allowed:
            return f"Error: use of '{name}' is not allowed"
    return str(eval(code, {}, allowed))  # noqa: S307


def _unsafe_gate2_only(expression: str) -> str:
    """假设的 ``calculate`` 变体，**仅有** 安全门 2（清空 builtins）。

    注意：没有静态名称检查，因此任何语法有效的表达式都会被编译并执行。``__builtins__={}`` 是唯一的防线。
    """
    allowed = {"abs": abs, "round": round, "min": min, "max": max}
    code = compile(expression, "<gate2-only>", "eval")
    try:
        return str(eval(code, {"__builtins__": {}}, allowed))  # noqa: S307
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def demo_ablation() -> None:
    _banner("第 4 部分 — 消融实验：为什么需要两道安全门")

    print("\n  场景 A — 开发者不小心将 '__import__' 加入了白名单。\n  仅靠安全门 2 还能救我们吗？")
    evil = "__import__('os').system('echo should-not-run')"

    # 仅安全门 1，但白名单配置错误：
    misconfigured_allowed = {"__import__": __import__}
    code = compile(evil, "<ablation>", "eval")
    all_names_allowed = all(n in misconfigured_allowed for n in code.co_names)
    print(f"    co_names 全部在白名单中? {all_names_allowed}  -> 安全门 1 会放行")

    # 安全门 2 是最后一道防线：
    try:
        eval(code, {"__builtins__": {}}, misconfigured_allowed)  # noqa: S307
        print("    安全门 2 裁决: 已执行（沙箱被突破！）")
    except Exception as e:
        print(f"    安全门 2 裁决: 已拦截 — {type(e).__name__}: {e}")

    print("\n  场景 B — 开发者忘记设置 __builtins__={}。\n  仅靠安全门 1 还能救我们吗？")
    evil = "__import__('os').system('echo should-not-run')"
    result = _unsafe_gate1_only(evil)
    print(f"    安全门 1 裁决: {result}")

    print("\n  场景 C — 开发者忘记了名称白名单。\n  仅靠安全门 2 能否阻止字面量属性链攻击？")
    evil = "(1).__class__.__bases__[0].__subclasses__()"
    result = _unsafe_gate2_only(evil)
    print(f"    安全门 2 裁决: {result}")


# ---------------------------------------------------------------------------
# 第 5 部分 — 要点总结
# ---------------------------------------------------------------------------


def demo_summary() -> None:
    _banner("第 5 部分 — 要点总结")
    print(
        """
  * 安全门 1（compile + co_names 白名单）— 语法级过滤。
      速度快，捕获属性链（__class__、__bases__ 等）、import名称，以及任何对非预期符号的引用。

  * 安全门 2（eval 配合 __builtins__={}）— 运行时隔离。
      即使攻击者将一个被禁名称绕过安全门 1（白名单配置错误、未来补丁、未知漏洞），空的 builtins 命名空间确保没有任何有意义的函数可供调用。

  两道门单独均不充分。两者结合产生"纵深防御"：攻击者必须在一个载荷中同时击败静态过滤器和运行时沙箱。术语来说：这是最小权限原则在 Agent 工具上的应用。
        """.rstrip()
    )


def main() -> None:
    demo_happy_path()
    demo_attacks()
    demo_garbage()
    demo_ablation()
    demo_summary()


if __name__ == "__main__":
    main()

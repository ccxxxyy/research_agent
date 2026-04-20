"""Demonstrate the two-gate sandbox that protects ``calculate`` from code injection.

Run:
    uv run python scripts/demo_calculator_sandbox.py

This script is NOT a unit test — it is a narrated demo meant to be read
alongside the source. It walks through the attacker's mental model:

    Attacker POV:  "calculate() uses eval. Surely I can escape."
    Defender POV:  "Two gates. Show me one that gets through."

For every attempt we print:
  1. the raw payload
  2. the compile-time ``co_names`` tuple (Gate 1's evidence)
  3. the return value from ``calculate`` (the defender's verdict)

We also run two "what if" ablations at the end: what if we had ONLY
Gate 1? ONLY Gate 2? — to make the layered-defence argument tangible.
"""

from __future__ import annotations

from research_agent.tools.native import calculate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inspect_co_names(expression: str) -> tuple[str, ...]:
    """Return the names the Python compiler would resolve at runtime.

    This is exactly what Gate 1 inspects inside ``calculate``. By printing
    it for each probe, the reader can see WHY a payload was rejected.
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
    verdict = "BLOCKED" if result.startswith("Error:") else "ALLOWED"
    print(f"\n  [{verdict}] {label}")
    print(f"    payload   : {expression!r}")
    print(f"    co_names  : {names}")
    print(f"    returned  : {result}")


# ---------------------------------------------------------------------------
# Part 1 — happy path (baseline: legitimate math still works)
# ---------------------------------------------------------------------------

def demo_happy_path() -> None:
    _banner("Part 1 — Legitimate math (should all ALLOW)")

    _run("basic arithmetic",        "2 + 3 * 4")
    _run("whitelisted builtin",     "round(3.14159, 2)")
    _run("whitelisted math module", "sqrt(16) + pi")


# ---------------------------------------------------------------------------
# Part 2 — attack catalogue (every one MUST be blocked)
# ---------------------------------------------------------------------------

def demo_attacks() -> None:
    _banner("Part 2 — Attack catalogue (should all BLOCK)")

    _run(
        "classic RCE via __import__",
        "__import__('os').system('echo pwned')",
    )
    _run(
        "filesystem read via open",
        "open('/etc/passwd').read()",
    )
    _run(
        "environ dump via builtins chain",
        "__builtins__['__import__']('os').environ",
    )
    _run(
        "subclass walk (the classic sandbox break)",
        "(1).__class__.__bases__[0].__subclasses__()",
    )
    _run(
        "dynamic exec",
        "exec('import os; os.system(\"id\")')",
    )
    _run(
        "nested eval",
        "eval(\"1+1\")",
    )
    _run(
        "globals inspection",
        "globals()",
    )
    _run(
        "attribute walk starting from a literal",
        "().__class__.__mro__",
    )


# ---------------------------------------------------------------------------
# Part 3 — syntax-invalid payloads (rejected before either gate)
# ---------------------------------------------------------------------------

def demo_garbage() -> None:
    _banner("Part 3 — Malformed payloads (SyntaxError handling)")

    _run("double operator", "2 ++ ** 3")
    _run("unclosed paren",  "((1+2)")


# ---------------------------------------------------------------------------
# Part 4 — ablation: WHY we need both gates
# ---------------------------------------------------------------------------

def _unsafe_gate1_only(expression: str) -> str:
    """Hypothetical ``calculate`` variant with ONLY Gate 1 (name whitelist).

    Notice: ``__builtins__`` is NOT cleared, so any name that slips through
    the whitelist resolves to a real builtin.
    """
    allowed = {"abs": abs, "round": round, "min": min, "max": max}
    code = compile(expression, "<gate1-only>", "eval")
    for name in code.co_names:
        if name not in allowed:
            return f"Error: use of '{name}' is not allowed"
    return str(eval(code, {}, allowed))  # noqa: S307


def _unsafe_gate2_only(expression: str) -> str:
    """Hypothetical ``calculate`` variant with ONLY Gate 2 (empty builtins).

    Notice: there is NO static name check, so any syntactically valid
    expression is compiled and evaluated. ``__builtins__={}`` is the only
    line of defence.
    """
    allowed = {"abs": abs, "round": round, "min": min, "max": max}
    code = compile(expression, "<gate2-only>", "eval")
    try:
        return str(eval(code, {"__builtins__": {}}, allowed))  # noqa: S307
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def demo_ablation() -> None:
    _banner("Part 4 — Ablation: why we need BOTH gates")

    print(
        "\n  Scenario A — developer accidentally adds '__import__' to the"
        "\n  whitelist. Would Gate 2 alone still save us?"
    )
    evil = "__import__('os').system('echo should-not-run')"

    # Gate 1 only, but with a misconfigured whitelist:
    misconfigured_allowed = {"__import__": __import__}
    code = compile(evil, "<ablation>", "eval")
    all_names_allowed = all(n in misconfigured_allowed for n in code.co_names)
    print(f"    co_names fully allow-listed? {all_names_allowed}  -> Gate 1 WOULD PASS")

    # Gate 2 is our last line of defence:
    try:
        eval(code, {"__builtins__": {}}, misconfigured_allowed)  # noqa: S307
        print("    Gate 2 verdict: RAN (sandbox breached!)")
    except Exception as e:
        print(f"    Gate 2 verdict: BLOCKED — {type(e).__name__}: {e}")

    print(
        "\n  Scenario B — developer forgets to set __builtins__={}. Would"
        "\n  Gate 1 alone still save us?"
    )
    evil = "__import__('os').system('echo should-not-run')"
    result = _unsafe_gate1_only(evil)
    print(f"    Gate 1 verdict: {result}")

    print(
        "\n  Scenario C — developer forgets the name whitelist. Would"
        "\n  Gate 2 alone still save us for a bare-literal attack chain?"
    )
    evil = "(1).__class__.__bases__[0].__subclasses__()"
    result = _unsafe_gate2_only(evil)
    print(f"    Gate 2 verdict: {result}")


# ---------------------------------------------------------------------------
# Part 5 — the takeaway
# ---------------------------------------------------------------------------

def demo_summary() -> None:
    _banner("Part 5 — Takeaway")
    print(
        """
  * Gate 1  (compile + co_names whitelist)  — syntactic filter.
      Fast, catches attribute chains (__class__, __bases__, ...),
      import names, and any reference to an unexpected symbol.

  * Gate 2  (eval with __builtins__={})     — runtime isolation.
      Even if an attacker slips a forbidden name past Gate 1
      (misconfigured whitelist, future patch, unknown bug),
      the empty builtins namespace ensures there is nothing
      meaningful to call.

  Neither gate is individually sufficient. Both together produce
  "defence in depth": an attacker must defeat both a static filter
  AND a runtime sandbox in a single payload. In interview terms:
  this is the principle of least privilege applied to agent tools.
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

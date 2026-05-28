"""Prompt 注入检测与输出安全过滤。

参考 OWASP Top 10 for Agentic AI (2025/2026)：
- LLM01: Prompt Injection (直接/间接)
- LLM02: Insecure Output Handling

本模块提供轻量级、基于规则的检测层（不依赖外部 LLM 调用），适合作为请求管道的第一道快速防线。
对于需要更高精度的场景，可在此基础上叠加基于 LLM 的二次验证。

设计原则
--------
- 快速：纯正则 + 字符串匹配，微秒级响应
- 低误报：模式尽量精准，避免误杀正常金融术语
- 可扩展：新模式只需添加到对应列表即可
- 可观测：每次检测返回结构化结果，含触发规则名
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class ThreatLevel(StrEnum):
    """威胁等级。"""

    SAFE = "safe"
    SUSPICIOUS = "suspicious"  # 可能误报，建议日志记录但放行
    BLOCKED = "blocked"        # 高置信度注入，应拦截


@dataclass(frozen=True)
class InputVerdict:
    """输入检测结果。"""

    level: ThreatLevel
    triggered_rules: tuple[str, ...] = ()
    sanitized_input: str | None = None

    @property
    def is_safe(self) -> bool:
        return self.level == ThreatLevel.SAFE


@dataclass(frozen=True)
class OutputVerdict:
    """输出检测结果。"""

    level: ThreatLevel
    triggered_rules: tuple[str, ...] = ()
    leaked_content: tuple[str, ...] = ()

    @property
    def is_safe(self) -> bool:
        return self.level == ThreatLevel.SAFE


@dataclass
class _Rule:
    name: str
    pattern: re.Pattern[str]
    level: ThreatLevel


def _compile_rules(raw: list[tuple[str, str, ThreatLevel]]) -> list[_Rule]:
    return [
        _Rule(name=name, pattern=re.compile(pat, re.IGNORECASE | re.DOTALL), level=level)
        for name, pat, level in raw
    ]


_INPUT_RULES_RAW: list[tuple[str, str, ThreatLevel]] = [
    # --- 直接注入：指令覆盖 ---
    (
        "ignore_instructions",
        r"(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|above|prior|earlier|system)\s+(?:instructions?|prompts?|rules?|context)",
        ThreatLevel.BLOCKED,
    ),
    (
        "new_instructions",
        r"(?:new|updated|revised)\s+(?:instructions?|system\s+prompt|rules?)[\s:]+",
        ThreatLevel.BLOCKED,
    ),
    # --- 角色劫持 ---
    (
        "role_hijack",
        r"(?:you\s+are\s+now|act\s+as|pretend\s+(?:to\s+be|you(?:'re|\s+are))|role[:\s]+|switch\s+(?:to\s+)?role)",
        ThreatLevel.BLOCKED,
    ),
    # --- 系统提示词提取 ---
    (
        "system_prompt_extraction",
        r"(?:(?:print|show|display|reveal|output|repeat|echo)\s+(?:your|the)\s+(?:system|initial|original)\s+(?:prompt|instructions?|message))",
        ThreatLevel.BLOCKED,
    ),
    (
        "system_prompt_extraction_v2",
        r"(?:what\s+(?:is|are)\s+your\s+(?:system|initial)\s+(?:prompt|instructions?))",
        ThreatLevel.SUSPICIOUS,
    ),
    # --- 越狱模板 ---
    (
        "jailbreak_dan",
        r"(?:DAN|do\s+anything\s+now|developer\s+mode|god\s+mode)",
        ThreatLevel.BLOCKED,
    ),
    (
        "jailbreak_delimiter",
        r"(?:\[/?SYSTEM\]|\[/?INST\]|<\|(?:im_start|im_end|system)\|>|<<SYS>>)",
        ThreatLevel.BLOCKED,
    ),
    # --- 间接注入标记 ---
    (
        "indirect_injection_marker",
        r"(?:IMPORTANT\s*(?:NEW\s+)?INSTRUCTION|BEGIN\s+INJECTION|INJECTED\s+PROMPT)",
        ThreatLevel.BLOCKED,
    ),
    # --- 编码绕过尝试 ---
    (
        "encoding_bypass",
        r"(?:base64|rot13|hex\s+encode|unicode\s+escape|url\s+encode).*(?:decode|convert|translate)",
        ThreatLevel.SUSPICIOUS,
    ),
]

_OUTPUT_RULES_RAW: list[tuple[str, str, ThreatLevel]] = [
    (
        "system_prompt_leak",
        r"(?:system\s+prompt|initial\s+instructions?)\s*(?:is|are|:)\s*.{20,}",
        ThreatLevel.BLOCKED,
    ),
    (
        "credential_leak",
        r"(?:(?:api[_\s]?key|secret|password|token)\s*[:=]\s*['\"]?\w{8,})",
        ThreatLevel.BLOCKED,
    ),
    (
        "internal_path_leak",
        r"(?:/(?:home|root|var|etc|usr)/\S{5,}|[A-Z]:\\\\(?:Users|Windows)\\\\\S{5,})",
        ThreatLevel.SUSPICIOUS,
    ),
]


class PromptGuard:
    """Prompt 注入检测器。

    Usage::

        guard = PromptGuard()
        verdict = guard.check_input(user_message)
        if not verdict.is_safe:
            logger.warning(f"Blocked input: {verdict.triggered_rules}")
            return error_response(...)

        # ... 正常处理 ...

        output_verdict = guard.check_output(llm_response, system_prompt)
        if not output_verdict.is_safe:
            return sanitized_response(...)
    """

    def __init__(
        self,
        *,
        extra_input_patterns: Sequence[tuple[str, str, ThreatLevel]] | None = None,
        extra_output_patterns: Sequence[tuple[str, str, ThreatLevel]] | None = None,
        system_prompt_fingerprints: Sequence[str] | None = None,
    ) -> None:
        raw_input = list(_INPUT_RULES_RAW)
        if extra_input_patterns:
            raw_input.extend(extra_input_patterns)
        self._input_rules = _compile_rules(raw_input)

        raw_output = list(_OUTPUT_RULES_RAW)
        if extra_output_patterns:
            raw_output.extend(extra_output_patterns)
        self._output_rules = _compile_rules(raw_output)

        self._system_fingerprints: tuple[str, ...] = tuple(
            system_prompt_fingerprints or []
        )

    def check_input(self, text: str) -> InputVerdict:
        """检测用户输入是否包含 prompt 注入尝试。"""
        triggered: list[str] = []
        max_level = ThreatLevel.SAFE

        for rule in self._input_rules:
            if rule.pattern.search(text):
                triggered.append(rule.name)
                if _level_severity(rule.level) > _level_severity(max_level):
                    max_level = rule.level

        return InputVerdict(
            level=max_level,
            triggered_rules=tuple(triggered),
        )

    def check_output(
        self,
        text: str,
        system_prompt: str | None = None,
    ) -> OutputVerdict:
        """检测 LLM 输出是否泄漏敏感信息。"""
        triggered: list[str] = []
        leaked: list[str] = []
        max_level = ThreatLevel.SAFE

        for rule in self._output_rules:
            match = rule.pattern.search(text)
            if match:
                triggered.append(rule.name)
                leaked.append(match.group(0)[:80])
                if _level_severity(rule.level) > _level_severity(max_level):
                    max_level = rule.level

        if system_prompt and len(system_prompt) > 20:
            chunks = _extract_fingerprint_chunks(system_prompt)
            for chunk in chunks:
                if chunk.lower() in text.lower():
                    triggered.append("system_prompt_verbatim_leak")
                    leaked.append(chunk[:80])
                    max_level = ThreatLevel.BLOCKED
                    break

        for fp in self._system_fingerprints:
            if fp.lower() in text.lower():
                triggered.append("system_fingerprint_match")
                leaked.append(fp[:80])
                max_level = ThreatLevel.BLOCKED
                break

        return OutputVerdict(
            level=max_level,
            triggered_rules=tuple(triggered),
            leaked_content=tuple(leaked),
        )


def _level_severity(level: ThreatLevel) -> int:
    return {ThreatLevel.SAFE: 0, ThreatLevel.SUSPICIOUS: 1, ThreatLevel.BLOCKED: 2}[level]


def _extract_fingerprint_chunks(prompt: str, min_len: int = 30) -> list[str]:
    """从系统提示词中提取有辨识度的片段，用于检测逐字泄漏。"""
    sentences = re.split(r'[。.!！?\?\n]', prompt)
    return [s.strip() for s in sentences if len(s.strip()) >= min_len]

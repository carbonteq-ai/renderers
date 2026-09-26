"""Tool and reasoning parsers for model families Posttrain trains (CarbonTeq fork).

Moved from Verifiers' ``renderer_extensions`` so the renderer owns every model
output format. Registered in ``renderers.parsers`` under ``lfm2`` and ``k2-ifm``.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from renderers.base import ParsedToolCall, ToolCallParseStatus
from renderers.pythonic_repair import repair_candidates, restore_reserved_kwarg_names


def _parse_pythonic_calls(raw: str) -> tuple[ast.expr, str] | None:
    """Parse a pythonic call list, applying vLLM's repairs if it is invalid.

    Serving parses the same text with these repairs, so training must accept
    the same calls or an episode that evaluation scores is lost in training.
    Returns the expression and the text it was parsed from.
    """

    for text in (raw, *repair_candidates(raw)):
        try:
            return ast.parse(text, mode="eval").body, text
        except (SyntaxError, ValueError):
            continue
    return None


class LFM2ToolParser:
    """Parse LFM2's ``[python_call(...)]`` tool-call token block.

    LFM emits a Python-expression surface syntax, but parsing is deliberately
    data-only: only a list of named calls, keyword arguments, and
    ``ast.literal_eval`` values are accepted.  No sampled code is executed.
    """

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._start = self._token_id("<|tool_call_start|>")
        self._end = self._token_id("<|tool_call_end|>")

    def _token_id(self, token: str) -> int | None:
        value = self._tokenizer.convert_tokens_to_ids(token)
        unknown = getattr(self._tokenizer, "unk_token_id", None)
        return value if isinstance(value, int) and value != unknown else None

    def extract(self, token_ids: list[int]) -> tuple[list[int], list[ParsedToolCall]]:
        if self._start is None or self._start not in token_ids:
            return token_ids, []
        start = token_ids.index(self._start)
        if self._end is None or self._end not in token_ids[start + 1 :]:
            raw = self._tokenizer.decode(
                token_ids[start + 1 :], skip_special_tokens=False
            )
            return token_ids[:start], [
                ParsedToolCall(
                    raw=raw,
                    token_span=(start, len(token_ids)),
                    status=ToolCallParseStatus.UNCLOSED_BLOCK,
                )
            ]
        end = token_ids.index(self._end, start + 1)
        raw = self._tokenizer.decode(
            token_ids[start + 1 : end], skip_special_tokens=False
        ).strip()
        span = (start, end + 1)
        parsed = _parse_pythonic_calls(raw)
        if parsed is None:
            return token_ids[:start], [
                ParsedToolCall(
                    raw=raw,
                    token_span=span,
                    status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                )
            ]
        expression, source = parsed
        if not isinstance(expression, ast.List) or not expression.elts:
            return token_ids[:start], [
                ParsedToolCall(
                    raw=raw,
                    token_span=span,
                    status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                )
            ]
        calls: list[ParsedToolCall] = []
        for item in expression.elts:
            if (
                not isinstance(item, ast.Call)
                or not isinstance(item.func, ast.Name)
                or item.args
                or any(keyword.arg is None for keyword in item.keywords)
            ):
                calls.append(
                    ParsedToolCall(
                        raw=ast.get_source_segment(source, item) or raw,
                        token_span=span,
                        status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                    )
                )
                continue
            keys = [str(keyword.arg) for keyword in item.keywords]
            if len(keys) != len(set(keys)):
                calls.append(
                    ParsedToolCall(
                        raw=ast.get_source_segment(source, item) or raw,
                        name=item.func.id,
                        token_span=span,
                        status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                    )
                )
                continue
            try:
                arguments = restore_reserved_kwarg_names(
                    {
                        str(keyword.arg): ast.literal_eval(keyword.value)
                        for keyword in item.keywords
                    }
                )
            except (TypeError, ValueError):
                calls.append(
                    ParsedToolCall(
                        raw=ast.get_source_segment(source, item) or raw,
                        name=item.func.id,
                        token_span=span,
                        status=ToolCallParseStatus.INVALID_JSON,
                    )
                )
                continue
            calls.append(
                ParsedToolCall(
                    raw=ast.get_source_segment(source, item) or raw,
                    name=item.func.id,
                    arguments=arguments,
                    token_span=span,
                )
            )
        return [*token_ids[:start], *token_ids[end + 1 :]], calls


class K2IFMToolParser:
    """Parse K2-Horizon's IFM XML tool-call block from sampled tokens."""

    _call = re.compile(r"<ifm\|tool_call>(.*?)</ifm\|tool_call>", flags=re.DOTALL)
    _argument = re.compile(
        r"\s*<ifm\|arg_key>(.*?)</ifm\|arg_key>\s*"
        r"<ifm\|arg_value>(.*?)</ifm\|arg_value>",
        flags=re.DOTALL,
    )

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._start = self._marker("<ifm|tool_calls>")
        self._end = self._marker("</ifm|tool_calls>")

    def _marker(self, text: str) -> list[int]:
        return [
            int(token)
            for token in self._tokenizer.encode(text, add_special_tokens=False)
        ]

    @staticmethod
    def _find(tokens: list[int], marker: list[int], start: int = 0) -> int | None:
        if not marker:
            return None
        limit = len(tokens) - len(marker) + 1
        return next(
            (
                index
                for index in range(start, limit)
                if tokens[index : index + len(marker)] == marker
            ),
            None,
        )

    def extract(self, token_ids: list[int]) -> tuple[list[int], list[ParsedToolCall]]:
        start = self._find(token_ids, self._start)
        if start is None:
            return token_ids, []
        body_start = start + len(self._start)
        end = self._find(token_ids, self._end, body_start)
        if end is None:
            raw = self._tokenizer.decode(
                token_ids[body_start:], skip_special_tokens=False
            )
            return token_ids[:start], [
                ParsedToolCall(
                    raw=raw,
                    token_span=(start, len(token_ids)),
                    status=ToolCallParseStatus.UNCLOSED_BLOCK,
                )
            ]
        block_end = end + len(self._end)
        raw_block = self._tokenizer.decode(
            token_ids[body_start:end], skip_special_tokens=False
        )
        span = (start, block_end)
        calls: list[ParsedToolCall] = []
        matches = list(self._call.finditer(raw_block))
        if not matches:
            calls.append(
                ParsedToolCall(
                    raw=raw_block,
                    token_span=span,
                    status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                )
            )
        for match in matches:
            raw = match.group(1).strip()
            name, separator, remainder = raw.partition("\n")
            name = name.strip()
            if not separator or not name:
                calls.append(
                    ParsedToolCall(
                        raw=raw,
                        token_span=span,
                        status=ToolCallParseStatus.MISSING_NAME,
                    )
                )
                continue
            arguments: dict[str, Any] = {}
            position = 0
            valid = True
            for argument in self._argument.finditer(remainder):
                if remainder[position : argument.start()].strip():
                    valid = False
                    break
                key = argument.group(1).strip()
                value_text = argument.group(2).strip()
                if not key or key in arguments:
                    valid = False
                    break
                try:
                    arguments[key] = json.loads(value_text)
                except json.JSONDecodeError:
                    arguments[key] = value_text
                position = argument.end()
            if remainder[position:].strip() or not valid:
                calls.append(
                    ParsedToolCall(
                        raw=raw,
                        name=name,
                        token_span=span,
                        status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                    )
                )
                continue
            calls.append(
                ParsedToolCall(raw=raw, name=name, arguments=arguments, token_span=span)
            )
        return [*token_ids[:start], *token_ids[block_end:]], calls


class K2IFMReasoningParser:
    """Preserve K2's generated thinking field across tool-result turns."""

    _close = re.compile(r"</ifm\|(think|think_fast|think_faster)>")
    _open = re.compile(r"^\s*<ifm\|(think|think_fast|think_faster)>\s*")

    def __init__(self, tokenizer: Any) -> None:
        del tokenizer

    def extract(self, text: str) -> tuple[str | None, str]:
        close = self._close.search(text)
        if close is None:
            return None, text
        before = text[: close.start()]
        opened = self._open.match(before)
        reasoning = before[opened.end() :] if opened is not None else before
        return reasoning or None, text[close.end() :]

"""Renderers for model families Posttrain trains that upstream lacks (CarbonTeq fork).

Each family renders through its own tokenizer chat template, like
``DefaultRenderer``, but declares what the generic fallback cannot know: the
assistant header its generation prompt ends with, its reasoning markers, its
turn-end stop tokens and its tool-call parser. With those, parsing places
reasoning on token boundaries (``ParsedResponse.reasoning_tokens``), detects a
reasoning channel opened by the prompt, and never leaks a turn-end token into
content.
"""

from __future__ import annotations

from typing import Any, ClassVar

from renderers.base import (
    ChatTemplateTokenizer,
    Message,
    ParsedResponse,
    ParsedToolCall,
    RenderedTokens,
    ToolSpec,
)
from renderers.configs import DefaultRendererConfig
from renderers.default import DefaultRenderer, _strip_special_tokens
from renderers.parsers import get_tool_parser
from renderers.parsing import (
    _build_param_type_index,
    _extract_tool_names,
    _find,
    _parse_glm_tool_calls,
    _parse_xml_tool_calls,
    _strip_stop_tokens,
)
from renderers.reasoning import (
    _single_marker_id,
    prompt_ends_in_reasoning,
    scan_reasoning,
)


class MarkedReasoningRenderer(DefaultRenderer):
    """Template renderer for a family with atomic reasoning markers.

    Subclasses set the class attributes below. Both reasoning markers must be
    single tokens in the family's tokenizer, so the reasoning boundary is
    exact; construction fails otherwise.
    """

    assistant_prefix: ClassVar[str]
    """Text the generation prompt ends with before the model's first token."""

    turn_end_tokens: ClassVar[tuple[str, ...]] = ()
    """Tokens that end an assistant turn, in addition to ``eos_token``."""

    default_tool_parser: ClassVar[str | None] = None
    """Tool parser used when the config leaves ``tool_parser`` unset."""

    tool_call_start_marker: ClassVar[str | None] = None
    """Tool-call opener that ends a reasoning region left without its close marker.

    vLLM's parsers treat an atomic tool-call opener as an implicit reasoning end;
    without it, a call sampled before ``</think>`` is swallowed as reasoning. Used
    only when the marker is a single token; an explicit close marker still wins.
    """

    def __init__(
        self,
        tokenizer: ChatTemplateTokenizer,
        config: DefaultRendererConfig | None = None,
    ):
        super().__init__(tokenizer, config)
        if self._tool_parser is None and self.default_tool_parser is not None:
            self._tool_parser = get_tool_parser(self.default_tool_parser, tokenizer)
        self._open_marker, self._close_marker = self.reasoning_markers()
        self._open_id = _single_marker_id(tokenizer, self._open_marker)
        self._close_id = _single_marker_id(tokenizer, self._close_marker)
        if self._open_id is None or self._close_id is None:
            raise ValueError(
                f"{type(self).__name__} requires {self._open_marker!r} and "
                f"{self._close_marker!r} to be single tokens in this tokenizer"
            )
        stop_ids: list[int] = list(super().get_stop_token_ids())
        for token in self.turn_end_tokens:
            token_id = _single_marker_id(tokenizer, token)
            if token_id is None:
                raise ValueError(
                    f"{type(self).__name__} requires turn-end token {token!r}"
                )
            if token_id not in stop_ids:
                stop_ids.append(token_id)
        self._stop_ids = stop_ids
        self._tool_start_id = (
            _single_marker_id(tokenizer, self.tool_call_start_marker)
            if self.tool_call_start_marker is not None
            else None
        )

    def reasoning_markers(self) -> tuple[str, str]:
        """Open and close markers of this renderer's reasoning channel."""
        return "<think>", "</think>"

    def extract_tool_calls(
        self,
        answer_ids: list[int],
        *,
        offset: int,
        tools: list[ToolSpec] | None,  # noqa: ARG002 — the generic tool parser has no schema
    ) -> tuple[list[int], list[ParsedToolCall]]:
        """Split post-reasoning tokens into content tokens and tool calls.

        ``offset`` is the position of ``answer_ids[0]`` in the completion;
        returned token spans are completion positions.
        """
        if self._tool_parser is None:
            return answer_ids, []
        content_ids, tool_calls = self._tool_parser.extract(answer_ids)
        for call in tool_calls:
            if call.token_span is not None:
                start, end = call.token_span
                call.token_span = (start + offset, end + offset)
        return content_ids, tool_calls

    def get_stop_token_ids(self) -> list[int]:
        return list(self._stop_ids)

    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools: list[ToolSpec] | None = None,
        prompt_ids: list[int] | None = None,
    ) -> ParsedResponse:
        stop_ids = set(self._stop_ids)
        ids = _strip_stop_tokens(list(token_ids), stop_ids)
        prefilled = prompt_ends_in_reasoning(
            self._tokenizer,
            prompt_ids,
            open_marker=self._open_marker,
            close_marker=self._close_marker,
            stop_ids=stop_ids,
            assistant_prefix=self.assistant_prefix,
        )
        boundary = scan_reasoning(
            self._tokenizer,
            ids,
            prefilled=prefilled,
            assistant_prefix=self.assistant_prefix,
            open_id=self._open_id,
            close_id=self._close_id,
            open_marker=self._open_marker,
            close_marker=self._close_marker,
            tool_start_id=self._tool_start_id,
            tool_start_closes_reasoning=self._tool_start_id is not None,
        )
        if boundary.is_open:
            return ParsedResponse(
                content="",
                reasoning_content=boundary.text,
                reasoning_complete=False,
                reasoning_tokens=boundary.token_count,
            )
        # Content (and any tool call) starts right after the reasoning region.
        reasoning_end = boundary.token_count or 0
        answer_ids, tool_calls = self.extract_tool_calls(
            ids[reasoning_end:], offset=reasoning_end, tools=tools
        )
        text = self._tokenizer.decode(answer_ids, skip_special_tokens=False)
        return ParsedResponse(
            content=_strip_special_tokens(self._tokenizer, text),
            reasoning_content=boundary.text,
            tool_calls=tool_calls,
            reasoning_complete=True,
            reasoning_tokens=reasoning_end,
        )


class LFM25Renderer(MarkedReasoningRenderer):
    """LiquidAI LFM2.5: ``<think>`` reasoning and pythonic tool-call lists."""

    assistant_prefix = "<|im_start|>assistant\n"
    turn_end_tokens = ("<|im_end|>",)
    default_tool_parser = "lfm2"
    tool_call_start_marker = "<|tool_call_start|>"

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,  # noqa: ARG002 — tool results render without the tool list
    ) -> RenderedTokens | None:
        """Append tool results without re-tokenizing the sampled history.

        LFM's template writes a newline after a history turn's ``<|im_end|>``
        that a stop-terminated completion never contains, so a full re-render
        changes the final token boundary. Render only the new tool messages,
        drop their standalone BOS, and join them with the template's newline.
        Other extensions return ``None`` so the caller re-renders.
        """
        return bridge_lfm25_tool_cycle(
            self,
            self._tokenizer,
            previous_prompt_ids,
            previous_completion_ids,
            new_messages,
        )


class K2HorizonRenderer(MarkedReasoningRenderer):
    """IFM K2-Horizon: effort-specific ``<ifm|think*>`` channels, IFM XML tools."""

    assistant_prefix = "<|ifm|im_start|>assistant\n"
    turn_end_tokens = ("<|ifm|im_end|>",)
    default_tool_parser = "k2-ifm"
    tool_call_start_marker = "<ifm|tool_calls>"

    _CHANNELS: ClassVar[dict[str, str]] = {
        "high": "think",
        "medium": "think_fast",
        "low": "think_faster",
    }

    def reasoning_markers(self) -> tuple[str, str]:
        effort = (self.config.model_extra or {}).get("reasoning_effort", "high")
        channel = self._CHANNELS.get(effort)
        if channel is None:
            raise ValueError(
                f"K2-Horizon reasoning_effort must be one of {sorted(self._CHANNELS)}, "
                f"got {effort!r}"
            )
        return f"<ifm|{channel}>", f"</ifm|{channel}>"


class Nanbeige42Renderer(MarkedReasoningRenderer):
    """Nanbeige 4.2: ``<think>`` reasoning and Qwen3.5-style XML tool calls."""

    assistant_prefix = "<|im_start|>assistant\n"
    turn_end_tokens = ("<|im_end|>",)
    default_tool_parser = "qwen3.5"
    tool_call_start_marker = "<tool_call>"

    def extract_tool_calls(self, answer_ids, *, offset, tools):
        # Schema-aware like Qwen35Renderer: string parameters stay verbatim.
        tc_id = _single_marker_id(self._tokenizer, "<tool_call>")
        tc_end_id = _single_marker_id(self._tokenizer, "</tool_call>")
        start = _find(answer_ids, tc_id) if tc_id is not None else -1
        if start == -1 or tc_end_id is None:
            return super().extract_tool_calls(answer_ids, offset=offset, tools=tools)
        return answer_ids[:start], _parse_xml_tool_calls(
            self._tokenizer,
            answer_ids[start:],
            tc_id,
            tc_end_id,
            section_offset=offset + start,
            param_index=_build_param_type_index(tools),
        )


class Spark25Renderer(MarkedReasoningRenderer):
    """XHToken Spark X2.5: ``<think>`` reasoning and GLM-style tool calls."""

    assistant_prefix = "<|Bot|>"
    default_tool_parser = "glm"
    tool_call_start_marker = "<tool_call>"

    def extract_tool_calls(self, answer_ids, *, offset, tools):
        # Schema-aware like the GLM renderers: string parameters stay verbatim.
        ids = {
            marker: _single_marker_id(self._tokenizer, marker)
            for marker in (
                "<tool_call>",
                "</tool_call>",
                "<arg_key>",
                "</arg_key>",
                "<arg_value>",
                "</arg_value>",
            )
        }
        start = (
            _find(answer_ids, ids["<tool_call>"])
            if ids["<tool_call>"] is not None
            else -1
        )
        if start == -1 or None in ids.values():
            return super().extract_tool_calls(answer_ids, offset=offset, tools=tools)
        return answer_ids[:start], _parse_glm_tool_calls(
            self._tokenizer,
            answer_ids[start:],
            ids["<tool_call>"],
            ids["</tool_call>"],
            ids["<arg_key>"],
            ids["</arg_key>"],
            ids["<arg_value>"],
            ids["</arg_value>"],
            section_offset=offset + start,
            param_index=_build_param_type_index(tools),
            known_names=_extract_tool_names(tools),
        )


def bridge_lfm25_tool_cycle(
    renderer: Any,
    tokenizer: Any,
    previous_prompt_ids: list[int],
    previous_completion_ids: list[int],
    new_messages: list[Message],
) -> RenderedTokens | None:
    """Extend an LFM2.5 tool cycle without re-tokenizing sampled tokens."""
    if not new_messages or any(m.get("role") != "tool" for m in new_messages):
        return None
    if not previous_completion_ids:
        return None
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(bos_token_id, int) or not isinstance(eos_token_id, int):
        return None
    suffix = renderer.render(new_messages, tools=None, add_generation_prompt=True)
    if not suffix.token_ids or int(suffix.token_ids[0]) != bos_token_id:
        return None
    close_ids = [] if previous_completion_ids[-1] == eos_token_id else [eos_token_id]
    newline_ids = [int(v) for v in tokenizer.encode("\n", add_special_tokens=False)]
    if not newline_ids:
        return None
    prefix_length = (
        len(previous_prompt_ids)
        + len(previous_completion_ids)
        + len(close_ids)
        + len(newline_ids)
    )
    suffix_is_content = list(suffix.is_content[1:]) if suffix.is_content else []
    suffix_sampled = list(suffix.sampled_mask[1:]) if suffix.sampled_mask else []
    return RenderedTokens(
        token_ids=[
            *previous_prompt_ids,
            *previous_completion_ids,
            *close_ids,
            *newline_ids,
            *suffix.token_ids[1:],
        ],
        message_indices=[-1] * prefix_length + list(suffix.message_indices[1:]),
        sampled_mask=([False] * prefix_length + suffix_sampled)
        if suffix_sampled
        else [],
        is_content=([False] * prefix_length + suffix_is_content)
        if suffix_is_content
        else [],
        message_roles=list(suffix.message_roles),
        message_tool_names=list(suffix.message_tool_names),
    )


__all__ = [
    "K2HorizonRenderer",
    "LFM25Renderer",
    "MarkedReasoningRenderer",
    "Nanbeige42Renderer",
    "Spark25Renderer",
    "bridge_lfm25_tool_cycle",
]

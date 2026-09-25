"""``ParsedResponse.reasoning_tokens`` places reasoning on completion-token boundaries.

The count covers generated reasoning text and markers; the answer (content and
tool calls) starts right after it. Reuses the per-renderer thinking streams from
``test_reasoning_boundaries`` so every registered reasoning renderer is covered.
"""

import pytest
from test_reasoning_boundaries import (
    _CASES,
    _TOOL_CALLS,
    _encode,
    _renderer,
    _thinking_stream,
)

from renderers import create_renderer
from renderers.base import load_tokenizer
from renderers.configs import config_from_name

# Formats whose parsers do not report token boundaries yet. They must say so
# with None rather than a wrong number.
_UNREPORTED = {"gpt-oss", "inkling", "inkling-small"}
_REASONING = sorted(set(_CASES) - {"llama-3", "default"})


@pytest.mark.parametrize("name", sorted(set(_REASONING) - _UNREPORTED))
@pytest.mark.parametrize("stop", [False, True])
def test_unfinished_reasoning_counts_every_generated_token(name, stop):
    tok, renderer = _renderer(name)
    prompt, incomplete, _ = _thinking_stream(name, tok, renderer)
    sampled = incomplete + (renderer.get_stop_token_ids()[:1] if stop else [])
    result = renderer.parse_response(sampled, prompt_ids=prompt)
    assert result.reasoning_complete is False
    assert result.reasoning_tokens == len(incomplete)


@pytest.mark.parametrize("name", sorted(set(_REASONING) - _UNREPORTED))
@pytest.mark.parametrize("stop", [False, True])
def test_closed_reasoning_ends_exactly_where_the_answer_starts(name, stop):
    tok, renderer = _renderer(name)
    prompt, incomplete, closed = _thinking_stream(name, tok, renderer)
    sampled = incomplete + closed + (renderer.get_stop_token_ids()[:1] if stop else [])
    result = renderer.parse_response(sampled, prompt_ids=prompt)
    assert result.content == "Answer"
    count = result.reasoning_tokens
    assert count is not None
    assert len(incomplete) < count <= len(incomplete) + len(closed)
    answer = tok.decode(
        sampled[count : len(incomplete) + len(closed)], skip_special_tokens=False
    )
    assert answer.strip() == "Answer"


@pytest.mark.parametrize("name", sorted(set(_TOOL_CALLS) - _UNREPORTED))
def test_tool_opener_that_ends_reasoning_belongs_to_the_answer(name):
    tok, renderer = _renderer(name)
    prompt, incomplete, _ = _thinking_stream(name, tok, renderer)
    sampled = incomplete + _encode(tok, _TOOL_CALLS[name])
    result = renderer.parse_response(sampled, prompt_ids=prompt)
    assert result.reasoning_complete is True
    count = result.reasoning_tokens
    assert count is not None
    # Whitespace between the thought and the opener stays with the thought.
    assert len(incomplete) <= count <= result.tool_calls[0].token_span[0]
    assert (
        tok.decode(sampled[len(incomplete) : count], skip_special_tokens=False).strip()
        == ""
    )


@pytest.mark.parametrize("name", ["default", "llama-3"])
def test_formats_without_reasoning_report_zero(name):
    if name == "default":
        # DefaultRenderer rejects explicit thinking_retention, so build it plainly.
        tok = load_tokenizer(_CASES[name])
        renderer = create_renderer(tok, config_from_name(name))
    else:
        tok, renderer = _renderer(name)
    sampled = _encode(tok, "Answer") + renderer.get_stop_token_ids()[:1]
    result = renderer.parse_response(sampled)
    assert result.reasoning_content is None
    assert result.reasoning_tokens == 0


@pytest.mark.parametrize("name", sorted(_UNREPORTED & set(_CASES)))
def test_unreported_formats_say_so(name):
    tok, renderer = _renderer(name)
    prompt, incomplete, closed = _thinking_stream(name, tok, renderer)
    assert (
        renderer.parse_response(incomplete + closed, prompt_ids=prompt).reasoning_tokens
        is None
    )

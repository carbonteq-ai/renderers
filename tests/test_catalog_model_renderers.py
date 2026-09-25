"""Catalog model renderers (CarbonTeq fork): reasoning, tools and turn ends.

Each Posttrain catalog family must place reasoning on token boundaries,
recognize reasoning its generation prompt opened, parse its own tool-call
format against the tool schema, and stop at its turn-end token. Strings are
encoded whole so tokenizers that add a word-boundary marker to standalone
strings do not change the expected text.
"""

from functools import lru_cache

import pytest

from renderers import create_renderer
from renderers.base import MODEL_RENDERER_MAP, ToolCallParseStatus, load_tokenizer
from renderers.catalog_models import (
    K2HorizonRenderer,
    LFM25Renderer,
    Nanbeige42Renderer,
    Spark25Renderer,
)
from renderers.configs import K2HorizonRendererConfig

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look something up.",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["q"],
            },
        },
    }
]

# model id -> (renderer class, close marker, tool call text, turn end)
_CASES = {
    "LiquidAI/LFM2.5-1.2B-Thinking": (
        LFM25Renderer,
        "</think>",
        "<|tool_call_start|>[lookup(q='x y', limit=3)]<|tool_call_end|>",
        "<|im_end|>",
    ),
    "LiquidAI/LFM2.5-2.6B": (
        LFM25Renderer,
        "</think>",
        "<|tool_call_start|>[lookup(q='x y', limit=3)]<|tool_call_end|>",
        "<|im_end|>",
    ),
    "Nanbeige/Nanbeige4.2-3B": (
        Nanbeige42Renderer,
        "</think>",
        "\n\n<tool_call>\n<function=lookup>\n<parameter=q>\nx y\n</parameter>\n"
        "<parameter=limit>\n3\n</parameter>\n</function>\n</tool_call>",
        "<|im_end|>",
    ),
    "XHToken/Spark-X2.5-4B": (
        Spark25Renderer,
        "</think>",
        "<tool_call>lookup<arg_key>q</arg_key><arg_value>x y</arg_value>"
        "<arg_key>limit</arg_key><arg_value>3</arg_value></tool_call>",
        "<｜end▁of▁sentence｜>",
    ),
    "IFM/K2-Horizon-7B": (
        K2HorizonRenderer,
        "</ifm|think>",
        "<ifm|tool_calls><ifm|tool_call>lookup\n<ifm|arg_key>q</ifm|arg_key>"
        '<ifm|arg_value>"x y"</ifm|arg_value><ifm|arg_key>limit</ifm|arg_key>'
        "<ifm|arg_value>3</ifm|arg_value></ifm|tool_call></ifm|tool_calls>",
        "<|ifm|im_end|>",
    ),
}


@lru_cache(None)
def _load(model: str):
    tok = load_tokenizer(model)
    renderer = create_renderer(tok)
    prompt = renderer.render_ids(
        [{"role": "user", "content": "Find x y."}],
        tools=_TOOLS,
        add_generation_prompt=True,
    )
    return tok, renderer, prompt


def _ids(tok, text: str) -> list[int]:
    return list(tok.encode(text, add_special_tokens=False))


def _thought(model: str, text: str) -> str:
    """Prefix the open marker unless the generation prompt already opened it."""
    tok, renderer, prompt = _load(model)
    open_marker = _CASES[model][1].replace("</", "<")
    opened = (
        tok.decode(prompt, skip_special_tokens=False).rstrip().endswith(open_marker)
    )
    return text if opened else open_marker + text


@pytest.mark.parametrize("model", sorted(_CASES))
def test_catalog_model_maps_to_its_renderer(model):
    cls = _CASES[model][0]
    _, renderer, _ = _load(model)
    assert type(renderer) is cls
    assert MODEL_RENDERER_MAP[model] == renderer.config.name


def test_gemma4_12b_uses_the_gemma4_renderer():
    assert MODEL_RENDERER_MAP["google/gemma-4-12B-it"] == "gemma4"


@pytest.mark.parametrize("model", sorted(_CASES))
def test_prompt_opened_thought_then_tool_call(model):
    _, close, call_text, turn_end = _CASES[model]
    tok, renderer, prompt = _load(model)
    sampled = _ids(
        tok, _thought(model, "plan the lookup") + close + call_text + turn_end
    )
    parsed = renderer.parse_response(sampled, prompt_ids=prompt, tools=_TOOLS)
    assert parsed.reasoning_content.strip() == "plan the lookup"
    assert parsed.reasoning_complete is True
    assert parsed.content.strip() == ""
    [call] = parsed.tool_calls
    assert (call.name, call.arguments, call.status) == (
        "lookup",
        {"q": "x y", "limit": 3},
        ToolCallParseStatus.OK,
    )
    count = parsed.reasoning_tokens
    thought = tok.decode(sampled[:count], skip_special_tokens=False)
    assert thought.endswith(close)
    assert call.token_span[0] >= count
    # The turn-end token is a stop token: never content, never reasoning.
    assert turn_end not in parsed.content
    assert count < len(sampled) - 1


@pytest.mark.parametrize("model", sorted(_CASES))
def test_thought_cut_off_by_the_length_limit(model):
    tok, renderer, prompt = _load(model)
    sampled = _ids(tok, _thought(model, "still planning the lookup and then"))
    parsed = renderer.parse_response(sampled, prompt_ids=prompt, tools=_TOOLS)
    assert parsed.reasoning_complete is False
    assert parsed.content == ""
    assert parsed.tool_calls == []
    assert parsed.reasoning_tokens == len(sampled)


@pytest.mark.parametrize("model", sorted(_CASES))
def test_answer_after_the_thought_and_late_markers_stay_content(model):
    _, close, _, turn_end = _CASES[model]
    tok, renderer, prompt = _load(model)
    open_marker = close.replace("</", "<")
    tail = "Done." + open_marker + "not reasoning" + close + "still content"
    sampled = _ids(tok, _thought(model, "plan") + close + tail + turn_end)
    parsed = renderer.parse_response(sampled, prompt_ids=prompt, tools=_TOOLS)
    assert parsed.reasoning_content.strip() == "plan"
    assert parsed.content == tail
    assert tok.decode(
        sampled[parsed.reasoning_tokens :], skip_special_tokens=False
    ).startswith("Done.")


@pytest.mark.parametrize("model", sorted(_CASES))
def test_self_contained_completion_with_explicit_thought(model):
    _, close, _, _ = _CASES[model]
    tok, renderer, _ = _load(model)
    open_marker = close.replace("</", "<")
    sampled = _ids(tok, open_marker + "plan" + close + "Answer")
    parsed = renderer.parse_response(sampled)
    assert parsed.reasoning_content.strip() == "plan"
    assert parsed.content == "Answer"
    assert parsed.reasoning_tokens == len(sampled) - len(_ids(tok, "Answer"))


@pytest.mark.parametrize(
    "effort,channel",
    [("high", "think"), ("medium", "think_fast"), ("low", "think_faster")],
)
def test_k2_reasoning_effort_selects_the_channel(effort, channel):
    tok = load_tokenizer("IFM/K2-Horizon-7B")
    renderer = create_renderer(tok, K2HorizonRendererConfig(reasoning_effort=effort))
    prompt = renderer.render_ids(
        [{"role": "user", "content": "hi"}], add_generation_prompt=True
    )
    assert tok.decode(prompt).endswith(f"<ifm|{channel}>\n")
    sampled = _ids(tok, f"plan</ifm|{channel}>Answer<|ifm|im_end|>")
    parsed = renderer.parse_response(sampled, prompt_ids=prompt)
    assert (parsed.reasoning_content, parsed.content) == ("plan", "Answer")
    assert parsed.reasoning_tokens == len(_ids(tok, f"plan</ifm|{channel}>"))


def test_k2_rejects_an_unknown_reasoning_effort():
    tok = load_tokenizer("IFM/K2-Horizon-7B")
    with pytest.raises(ValueError, match="reasoning_effort"):
        create_renderer(tok, K2HorizonRendererConfig(reasoning_effort="max"))


@pytest.mark.parametrize(
    "model", ["LiquidAI/LFM2.5-1.2B-Thinking", "LiquidAI/LFM2.5-2.6B"]
)
def test_lfm_tool_results_extend_the_sampled_history(model):
    tok, renderer, prompt = _load(model)
    completion = _ids(
        tok,
        _thought(model, "plan")
        + "</think><|tool_call_start|>[lookup(q='x')]<|tool_call_end|><|im_end|>",
    )
    tool = [{"role": "tool", "content": "result", "tool_call_id": "c1"}]
    bridged = renderer.bridge_to_next_turn(prompt, completion, tool)
    assert bridged is not None
    assert bridged.token_ids[: len(prompt) + len(completion)] == prompt + completion
    assert not any(bridged.sampled_mask[len(prompt) + len(completion) :])
    # The bridge ends with the same generation prompt a full render would.
    full = renderer.render_ids(
        [
            {"role": "user", "content": "Find x y."},
            {"role": "assistant", "content": "", "reasoning_content": "plan"},
            *tool,
        ],
        tools=_TOOLS,
        add_generation_prompt=True,
    )
    assert bridged.token_ids[-3:] == full[-3:]
    # A new user query re-renders instead of bridging.
    user = [{"role": "user", "content": "next"}]
    assert renderer.bridge_to_next_turn(prompt, completion, user) is None

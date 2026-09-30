"""LFM SFT targets exclude inference-prefilled headers and separators."""

from pathlib import Path

import pytest
from transformers import AutoTokenizer

from renderers import LFM25RendererConfig, build_training_sample, create_renderer


@pytest.fixture(
    params=[
        ("LFM2.5-1.2B-Thinking", "95053d21d8e0b7ca99421a2127ae39c64f685ff3"),
        ("LFM2.5-2.6B", "654f9463ce32b05d0429d76fe1f580b27d4c1ac0"),
    ]
)
def lfm_renderer(request):
    model, revision = request.param
    path = (
        Path.home()
        / ".cache/huggingface/hub"
        / f"models--LiquidAI--{model}"
        / "snapshots"
        / revision
    )
    if not path.exists():
        pytest.skip(f"requires cached immutable {model} tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return tokenizer, create_renderer(tokenizer, LFM25RendererConfig())


@pytest.mark.parametrize(
    "content",
    [
        "Four.",
        "<think>check</think>Four.",
        "",
        "Literal <|im_start|>assistant inside the answer.",
    ],
)
def test_sampled_completion_equals_generation_suffix_through_stop(
    lfm_renderer, content
):
    tokenizer, renderer = lfm_renderer
    prompt = [
        {"role": "system", "content": "Be accurate."},
        {"role": "user", "content": "Two plus two?"},
    ]
    messages = [*prompt, {"role": "assistant", "content": content}]
    prefix = renderer.render_ids(prompt, add_generation_prompt=True)
    complete = renderer.render_ids(messages)
    # The 2.6B template adds a reasoning prefill on inference prompts even
    # when a curated full SFT answer contains no reasoning block.
    if tokenizer.decode(prefix).endswith("<think>") and not content.startswith(
        "<think>"
    ):
        prefix = prefix[:-1]
    rendered = renderer.render(messages, add_generation_prompt=True)
    stop = max(
        k
        for k in range(len(prefix), len(complete))
        if complete[k] in renderer.get_stop_token_ids()
    )
    assert rendered.sampled_mask == [
        len(prefix) <= k <= stop for k in range(len(rendered.token_ids))
    ]
    sample = build_training_sample(
        renderer,
        messages,
        role_to_mask=lambda message: message["role"] == "assistant",
        ensure_final_stop=True,
    )
    selected = [
        token
        for token, keep in zip(sample.token_ids, sample.loss_mask, strict=True)
        if keep
    ]
    assert selected == complete[len(prefix) : stop + 1]
    assert selected[-1] in renderer.get_stop_token_ids()
    if content == "Four.":
        assert tokenizer.decode(selected) == "Four.<|im_end|>"


@pytest.mark.parametrize("selected", [[2], [4], [2, 4]])
def test_multiturn_selection_excludes_tool_body_and_assistant_openers(
    lfm_renderer, selected
):
    tokenizer, renderer = lfm_renderer
    messages = [
        {"role": "system", "content": "SYSTEM_SENTINEL"},
        {"role": "user", "content": "USER_SENTINEL"},
        {
            "role": "assistant",
            "content": "FIRST_ANSWER",
            "tool_calls": [
                {
                    "id": "a",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"key":"x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "TOOL_SENTINEL"},
        {"role": "assistant", "content": "FINAL_ANSWER"},
    ]
    identities = {id(messages[k]) for k in selected}
    sample = build_training_sample(
        renderer, messages, role_to_mask=lambda message: id(message) in identities
    )
    text = tokenizer.decode(
        [
            token
            for token, keep in zip(sample.token_ids, sample.loss_mask, strict=True)
            if keep
        ]
    )
    assert all(
        marker not in text
        for marker in [
            "SYSTEM_SENTINEL",
            "USER_SENTINEL",
            "TOOL_SENTINEL",
            "<|im_start|>assistant",
        ]
    )
    assert ("FIRST_ANSWER" in text) == (2 in selected)
    assert ("FINAL_ANSWER" in text) == (4 in selected)


def test_inconsistent_generation_prefix_fails_explicitly(lfm_renderer, monkeypatch):
    _, renderer = lfm_renderer
    original = renderer._apply

    def inconsistent(messages, *, tools=None, add_generation_prompt=False):
        ids = original(
            messages, tools=tools, add_generation_prompt=add_generation_prompt
        )
        return [*ids, 123] if add_generation_prompt else ids

    monkeypatch.setattr(renderer, "_apply", inconsistent)
    with pytest.raises(ValueError, match="generation-prompt token prefix"):
        renderer.render(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        )


@pytest.mark.parametrize("content", ["Four.", "<think>check</think>Four."])
def test_assistant_only_sample_excludes_bos_and_injected_header(lfm_renderer, content):
    tokenizer, renderer = lfm_renderer
    sample = build_training_sample(
        renderer, [{"role": "assistant", "content": content}]
    )
    selected = [
        token
        for token, keep in zip(sample.token_ids, sample.loss_mask, strict=True)
        if keep
    ]
    expected = content + "<|im_end|>"
    if "2.6B" in tokenizer.name_or_path and content.startswith("<think>"):
        expected = expected.removeprefix("<think>")
    assert tokenizer.decode(selected) == expected

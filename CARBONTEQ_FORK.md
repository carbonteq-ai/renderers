# CarbonTeq AI renderers fork

This repository is a fork of
[`PrimeIntellect-ai/renderers`](https://github.com/PrimeIntellect-ai/renderers).
It makes the renderer the single owner of model output accounting for every
model in the Posttrain catalog. Posttrain consumes it through Verifiers and its
own TRL training path. The consumer record is `docs/tooling/renderers/README.md`
in the Posttrain repository (`carbonteq-ai/rl`); the implementation plan is
`docs/plan/renderers-fork-thinking-token-accounting.md` there.

## Status

Candidate, published to GitHub. The latest release is `carbonteq-v0.1.12.post1.dev3` (pre-release); see Releases.

Source-only sampled-mask correction, branch `codex/lfm-sampled-mask`, based on
ledger commit `d1458bf1a665278b05ac6e9ed0611abc78f953a8` (published dev2 behavior).
`LFM25Renderer.render` now populates sampled assistant spans from the exact
inference-prefilled header boundary through the emitted turn stop. Injected
assistant headers, BOS, separators, appended generation prompts, and
system/user/tool observations remain unscored. LFM2.5-2.6B prefills `<think>`
in the generation prompt; LFM2.5-1.2B-Thinking emits it itself. The mask
preserves that difference, supports plain curated SFT answers and assistant-only
samples, and rejects inconsistent template prefix boundaries explicitly.
It preserves serialized token IDs and parsing/bridge behavior. Other families
are unchanged. The existing incremental template renderer gets additional
prefix checks; preprocessing overhead on long histories remains to be measured.

`tests/test_lfm_sampled_mask.py` exercises immutable cached 1.2B/2.6B tokenizers
with plain, reasoning, empty, literal-role-marker, selective multi-turn,
assistant-only, and invalid-prefix cases. All 20 fail on the isolated published
dev2 wheel and pass on the candidate. The new cases plus existing LFM sampled
mask, build-helper, parser and bridge tests pass 59 cases; 320 unrelated cases
are deselected. Posttrain's renderer/SFT-validation integration passes 15
cases, including three new header/tool masking regressions. On a real 1.2B
three-step SFT probe, targets shrink from seven to three tokens and independent
loss/parameter gradients match; three DPO updates retain prior behavior.
This is target/numerical correctness evidence, not task convergence or native
Verifiers trajectory qualification.

Follow-up (2026-10-02): the template rewrites assistant turns that precede a
later user message (2.6B drops their reasoning), so a turn rendered as if last
is not a prefix of the full conversation. `render` now derives such a turn's
boundary from its historical form (the conversation through that turn plus an
empty user message, ended at the first turn stop), restoring the
`test_offsetless_tokenizers` reasoning-history case and adding
`test_reasoning_history_turn_is_masked_in_its_rewritten_form`. Released as
`0.1.12.post1.dev3` (see Releases).

## Distribution and remotes

CarbonTeq publishes the fork as `carbonteq-renderers`. The import package stays
`renderers`, so consumers change only their dependency name. The distinct name
keeps a public `renderers` release from replacing the fork during resolution.
Consumers must not depend on both distributions, because both install the
`renderers` import package.

Fork releases are versioned `<upstream base>.post<n>.dev<m>` and tagged
`carbonteq-v<version>`. `renderers/_version.py` is the single source of the
version (`[tool.hatch.version] path` in `pyproject.toml`). Upstream derives
versions from `renderers-v*` tags with hatch-vcs; the fork does not, so
untagged fork commits still install. Upstream's `.github/workflows/publish.yml`
triggers only on `renderers-v*` tags and PyPI trusted publishing, so it never
runs for fork releases. Fork releases follow Posttrain's retained-asset
publication route instead, not a fork-controlled workflow.

Expected remotes: `origin` is `https://github.com/carbonteq-ai/renderers.git`
and `upstream` is `https://github.com/PrimeIntellect-ai/renderers.git`.

## Upstream base

`PrimeIntellect-ai/renderers` `main` at
`20f2b38c` ("fix: treat tool-call openers as implicit reasoning ends (vLLM 0.26
parity) (#158)", 2026-09-24), which upstream's own versioning would call
`0.1.12.dev10`. Posttrain previously consumed upstream `0.1.12.dev3`
(`06bcf635`). The newer base includes #152 (unfinished reasoning survives
parsing and bridging), #153 (`ParsedResponse.reasoning_complete`) and #158,
which the accounting below depends on.

## Maintained delta

### Distribution identity

`pyproject.toml`, `renderers/_version.py`, `.gitignore`, `uv.lock`: the
distribution rename and version source described above. Rebase review: keep
the name and the version path; take upstream's dependency changes.

### Declared httpx dependency

`renderers/client.py` imports `httpx`, which upstream never declares; it
arrived through `openai`. `openai` 3.x depends on `httpx2` instead, so a clean
install of the wheel failed on `import renderers`. `pyproject.toml` declares
`httpx>=0.27`. Retire this delta when upstream declares it.

### Reasoning token accounting

Why: `ParsedResponse` reports reasoning text and whether it is complete, but not
which completion tokens it occupied. Consumers that need thinking-token counts
(training metrics, trace evidence, length shaping) otherwise re-derive them
with model-specific rules outside the renderer.

Behavior that must survive an upstream merge:

- `ReasoningBoundary.token_count` (`renderers/reasoning.py`): `scan_reasoning`
  reports how many scanned completion tokens belong to reasoning. That includes
  generated open/close markers and a generated assistant header before the
  region. A tool-call opener that ends the region is not included, and stop
  tokens never are. The count is 0 with no region and covers every scanned
  token while a region is open. Text-marker formats map character boundaries
  to tokens with `_tokens_through_char`. Channel formats (`initial_only=False`)
  sum their region spans.
- `ParsedResponse.reasoning_tokens` (`renderers/base.py`) carries that count.
  Every parser built on `scan_reasoning` passes it through
  (`renderers/parsing.py`, `renderers/default.py`, `renderers/kimi_k25.py`).
- Exceptions:
  - `renderers/gemma4.py` counts from its own first-thought cursor, captured
    before the tool-call loop, so the count matches its `reasoning_content`.
  - `parse_deepseek_v4` uses the DSML section start when a tool call ends the
    thought, because that opener is text around the atomic marker the scan
    anchors on.
  - `parse_llama_3` reports 0.
- Deliberately unsupported: `gpt-oss` (Harmony channels) and `inkling` report
  `None` until their parsers place reasoning on token boundaries.

Regression tests: `tests/test_reasoning_token_accounting.py`, which reuses
`tests/test_reasoning_boundaries.py`'s per-renderer thinking streams.

Conflict-sensitive areas on rebase: `scan_reasoning`'s return paths, every
`ParsedResponse(...)` construction in the parsers, and `Gemma4Renderer.parse_response`.
After a rebase, run the regression tests and search for any new
`ParsedResponse(` site that lacks `reasoning_tokens`.

### Catalog model renderers

Why: upstream has dedicated renderers for Qwen 3.5 and Gemma 4 only. The
other Posttrain catalog families fell back to `DefaultRenderer`, which assumes
Qwen's `<|im_start|>assistant\n` header, `<think>` markers and the tokenizer's
`eos_token` as the only stop. That fallback missed Spark's prompt-opened
thought (`<|Bot|>` header) and K2's `<ifm|think*>` channel entirely, and leaked
K2's `<|ifm|im_end|>` turn end into content. The LFM2 and K2 tool parsers lived
in Verifiers.

Behavior that must survive an upstream merge:

- `renderers/catalog_models.py`: `MarkedReasoningRenderer`, a
  `DefaultRenderer` subclass. Each family declares its assistant header,
  single-token reasoning markers, turn-end tokens and tool parser. Parsing
  detects reasoning opened by the prompt, counts `reasoning_tokens` exactly,
  and strips turn ends.
- The four family renderers:
  - `LFM25Renderer` (`lfm2.5`): `lfm2` pythonic tool calls, and the tool-cycle
    bridge (`bridge_lfm25_tool_cycle`) that keeps sampled history
    byte-identical when tool results are appended.
  - `K2HorizonRenderer` (`k2-horizon`): `reasoning_effort` selects the
    `<ifm|think>`, `<ifm|think_fast>` or `<ifm|think_faster>` channel.
    `k2-ifm` tool calls.
  - `Nanbeige42Renderer` (`nanbeige4.2`): schema-aware Qwen 3.5 XML tool calls.
  - `Spark25Renderer` (`spark2.5`): schema-aware GLM tool calls.
- `renderers/catalog_parsers.py`: `LFM2ToolParser`, `K2IFMToolParser` and
  `K2IFMReasoningParser`, moved from Verifiers and registered in
  `renderers/parsers.py`.
- Registration:
  - configs `LFM25RendererConfig`, `K2HorizonRendererConfig`,
    `Nanbeige42RendererConfig` and `Spark25RendererConfig` (subclasses of
    `DefaultRendererConfig`; template kwargs pass through) in
    `renderers/configs.py`, including the `RendererConfig` union;
  - entries in `RENDERER_REGISTRY` and `MODEL_RENDERER_MAP` in
    `renderers/base.py`;
  - `google/gemma-4-12B-it` (Gemma 4 Unified: image and audio encoders)
    mapped to `gemma4` for text only, and listed with 26B and 31B in
    `_EMPTY_THOUGHT_PREFILL_MODELS`, since it ships the same template
    revision. It is deliberately absent from `MULTIMODAL_MODELS`: its media
    processor path is not qualified.
- `renderers/client.py` `generate()` returns `reasoning_tokens` next to
  `reasoning_content`.
- A reasoning region left without its close marker ends at the family's
  single-token tool-call opener (`tool_call_start_marker`: LFM2.5
  `<|tool_call_start|>`, K2 `<ifm|tool_calls>`, Nanbeige and Spark
  `<tool_call>`), as vLLM's parsers and the upstream renderers do; an explicit
  close still wins. Without it a call sampled before `</think>` was swallowed as
  reasoning and never parsed: 28 of 520 LFM2.5-2.6B AutomationBench training
  episodes (Posttrain run `lfm26-vortex-v5-150-dspark-opt-20260926-r1`, updates
  41-56) ended on such a call.
- `LFM2ToolParser` applies the same repairs as the CarbonTeq vLLM fork's `lfm2`
  tool parser when a call is not valid Python: raw control characters and NUL
  inside strings are escaped, zero-padded integers (`month=07`) are
  normalized, a string broken by an unescaped quote of its own kind
  (`subject='Let's go'`) is closed at the only closer that parses, and
  keyword-named parameters (`from=`) are renamed and restored. The first
  rewrite that parses wins; ambiguous or unrecoverable text is still
  `MALFORMED_STRUCTURE`. `renderers/pythonic_repair.py` is an unchanged copy of
  those vLLM helpers (`vllm/tool_parsers/utils.py`) and must stay identical to
  it. Without them training dropped calls that serving accepts: LFM2.5-1.2B
  scored 0 in training on `simple.gmail_onboarding_welcome` (the call was
  dropped) and 1.0 on every evaluation attempt (Posttrain runs
  `lfm12-sampo-8gb-20260926-r6` and `lfm12-screen-8k-20260926-r2`).
- Like `DefaultRenderer`, these renderers reject an explicit
  `thinking_retention`. Only LFM bridges, and only tool results; other
  extensions re-render through the chat template.

Regression tests:

- `tests/test_catalog_model_renderers.py`: every catalog model's mapping,
  prompt-opened thought then tool call, a tool call ending an unclosed thought,
  cut-off thought, late markers,
  self-contained thought, the K2 efforts, the LFM bridge, the LFM tool calls
  vLLM repairs, and an unrecoverable call that stays malformed.
- The catalog models added to `tests/parity.py` `MODEL_CATALOG`, which runs the
  upstream parity and reasoning suites.
- In `tests/test_reasoning_boundaries.py`, `_TEMPLATE_RENDERED` exempts these
  renderers from the retention and user-turn bridge tests, as upstream exempts
  `default`.
- K2-Horizon is excluded from the generic assistant parity scenarios, because
  its template raises on assistant messages without a thinking field.

## Validation

    uv sync --python 3.13
    uv run ruff check renderers tests
    uv run pytest tests/test_reasoning_token_accounting.py -q
    uv run pytest tests/ -q

The tests download tokenizers from Hugging Face on first use.

## Rebase procedure

    git fetch upstream
    git rebase upstream/main
    # Resolve conflicts, keeping the distribution identity and
    # reasoning_tokens pass-through as described above.
    uv lock
    uv run pytest tests/ -q

Record the new base commit in this file and in Posttrain's
`docs/tooling/renderers/README.md`.

## Releases

- `0.1.12.post1.dev1`: tag `carbonteq-v0.1.12.post1.dev1` at fork commit
  `6aba28a8c9a597475addc2c123dd18b28a462766` (branch
  `carbonteq/thinking-token-accounting`), GitHub pre-release with retained
  assets. Wheel `carbonteq_renderers-0.1.12.post1.dev1-py3-none-any.whl`
  SHA-256 `2e3231784729b9177bfc25eb06e3a6a958f9ba420ff8e241c010266cc9422d0b`;
  sdist `carbonteq_renderers-0.1.12.post1.dev1.tar.gz` SHA-256
  `bf529fc910f66494770a96e0ee5ec986344ed5f8b48258b50a55b82baa876fa3`. Built
  with `uv build` from `git archive` of the tagged commit; `twine check`
  passed and a clean install imports and builds the LFM2.5 renderer.
- `0.1.12.post1.dev2`: tag `carbonteq-v0.1.12.post1.dev2` at fork commit
  `6f71261` (branch `carbonteq/tool-start-closes-reasoning`), GitHub
  pre-release with retained assets. Adds the tool-call opener ending an
  unclosed thought and the LFM2 pythonic tool-call repairs. Wheel
  `carbonteq_renderers-0.1.12.post1.dev2-py3-none-any.whl` SHA-256
  `fdc65e9ed1a8a2f877c127a3456834d5ded996f32a4c70fe22bebe615073a87e`; sdist
  `carbonteq_renderers-0.1.12.post1.dev2.tar.gz` SHA-256
  `2c83d94bd1fd81f5fdfe95cd8390ac1d5057355bc278bf14e642ab3f8b2c26bd`. Built
  with `uv build` from `git archive` of the tagged commit; `twine check`
  passed and a clean install parses a repaired LFM2.5 call. Published to
  `pypi.lan/carbonteq/dev` by Posttrain run 36259417685.
- `0.1.12.post1.dev3`: tag `carbonteq-v0.1.12.post1.dev3` at fork commit
  `7fe5d06b9840ef0e4c7d43419cbd7f9afc1b727a` (branch `codex/lfm-sampled-mask`),
  GitHub pre-release with retained assets. Adds the LFM sampled SFT mask
  correction and history-turn masking in the template-rewritten form. Wheel
  `carbonteq_renderers-0.1.12.post1.dev3-py3-none-any.whl` SHA-256
  `57dcc6f8ba2db7bec21c75d40177059f704bb817a772558435e4d34c262ab5eb`; sdist
  `carbonteq_renderers-0.1.12.post1.dev3.tar.gz` SHA-256
  `650954a172b93524e871f873fad0d5448bc9e578bd8da2fe3e46d48b9307aa39`. Built
  with `uv build` from `git archive` of the tagged commit; `twine check`
  passed; the full suite passes 11,834 cases. Published to
  `pypi.lan/carbonteq/dev` by Posttrain run 36996039659.

# CarbonTeq AI renderers fork

This repository is a fork of
[`PrimeIntellect-ai/renderers`](https://github.com/PrimeIntellect-ai/renderers).
It makes the renderer the single owner of model output accounting for every
model in the Posttrain catalog. Posttrain consumes it through Verifiers and its
own TRL training path. The consumer record is `docs/tooling/renderers/README.md`
in the Posttrain repository (`carbonteq-ai/rl`); the implementation plan is
`docs/plan/renderers-fork-thinking-token-accounting.md` there.

## Status

Candidate, unpublished. No `carbonteq-v*` release exists yet.

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

None yet.

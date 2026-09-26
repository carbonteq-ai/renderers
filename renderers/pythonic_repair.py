"""Repairs that let near-valid pythonic tool calls parse, shared with vLLM.

Copied unchanged from the CarbonTeq vLLM fork's ``vllm/tool_parsers/utils.py``
(Apache-2.0), which its ``lfm2`` tool parser applies when serving. Training
parses sampled tokens with this package instead, so without the same repairs a
call that evaluation accepts (``subject='Let's go'``, ``month=07``,
``from=...``) was dropped in training and the episode scored as if no tool was
called. Keep these functions identical to the vLLM copy.
"""

from __future__ import annotations

import ast
import keyword as _python_keyword
import warnings


def escape_ctrl_chars_in_strings(text: str) -> str:
    """Escape literal control chars inside string literals of pythonic text.

    Models emitting pythonic tool calls frequently place raw newlines inside a
    string argument (e.g. ``exec(command='line1\\nline2')`` written with a real
    line break). That is invalid Python — ``ast.parse`` fails with "unterminated
    string literal" — so the call would be dropped even though the intent is
    unambiguous. A NUL byte is worse: ``ast.parse`` rejects it anywhere in the
    source with ``ValueError``. Escaping ``\\n``/``\\r``/``\\t``/``\\x00`` only
    *inside* string literals makes the text parseable while preserving the
    argument value exactly (the escape sequences evaluate back to the original
    control chars).

    Text outside string literals is returned unchanged.
    """
    out: list[str] = []
    quote: str | None = None
    index, length = 0, len(text)
    while index < length:
        char = text[index]
        if quote is None:
            if char in {"'", '"'}:
                quote = char
            out.append(char)
        elif char == "\\" and index + 1 < length:
            out.append(char)
            out.append(text[index + 1])
            index += 2
            continue
        elif char == quote:
            quote = None
            out.append(char)
        elif char == "\n":
            out.append("\\n")
        elif char == "\r":
            out.append("\\r")
        elif char == "\t":
            out.append("\\t")
        elif char == "\x00":
            out.append("\\x00")
        else:
            out.append(char)
        index += 1
    return "".join(out)


_RESERVED_KW_SUFFIX = "_pyreservedkw_"


def rename_reserved_kwargs(text: str) -> tuple[str, bool]:
    """Rename Python-keyword parameter names so pythonic tool text parses.

    Tools legitimately name parameters ``from``, ``in``, ``class`` — but
    ``memory_get(from=1)`` is a Python ``SyntaxError``, so the whole call
    would be dropped. Rename ``from=`` to ``from_pyreservedkw_=`` (outside
    string literals only, and only in keyword-argument position: preceded by
    ``(`` or ``,`` and followed by a single ``=``), parse, then restore the
    original name with :func:`restore_reserved_kwarg_names`.

    Returns (rewritten_text, changed). Keyword *values* (``x=True``) and
    keywords inside string arguments are never touched.
    """
    out: list[str] = []
    quote: str | None = None
    changed = False
    last_sig = ""
    index, length = 0, len(text)
    while index < length:
        char = text[index]
        if quote is not None:
            out.append(char)
            if char == "\\" and index + 1 < length:
                out.append(text[index + 1])
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            out.append(char)
            last_sig = char
            index += 1
            continue
        if char.isalpha() or char == "_":
            end = index
            while end < length and (text[end].isalnum() or text[end] == "_"):
                end += 1
            name = text[index:end]
            look = end
            while look < length and text[look].isspace():
                look += 1
            if (
                _python_keyword.iskeyword(name)
                and look < length
                and text[look] == "="
                and (look + 1 >= length or text[look + 1] != "=")
                and last_sig in {"(", ","}
            ):
                out.append(name + _RESERVED_KW_SUFFIX)
                changed = True
            else:
                out.append(name)
            last_sig = name[-1]
            index = end
            continue
        out.append(char)
        if not char.isspace():
            last_sig = char
        index += 1
    return "".join(out), changed


def restore_reserved_kwarg_names(arguments: dict) -> dict:
    """Undo :func:`rename_reserved_kwargs` on a decoded arguments dict.

    Only keys that carry the rename suffix *and* whose stem is a Python
    keyword are restored, making this an exact inverse of the rename.
    """
    restored = {}
    for key, value in arguments.items():
        if (
            isinstance(key, str)
            and key.endswith(_RESERVED_KW_SUFFIX)
            and _python_keyword.iskeyword(key[: -len(_RESERVED_KW_SUFFIX)])
        ):
            restored[key[: -len(_RESERVED_KW_SUFFIX)]] = value
        else:
            restored[key] = value
    return restored


def normalize_leading_zero_ints(text: str) -> str:
    """Strip leading zeros from decimal integer literals so the text parses.

    Models emit zero-padded integers (``month=07``), which Python rejects
    ("leading zeros in decimal integer literals are not permitted"), so the
    whole call would be dropped. Rewrite ``07`` to ``7`` outside string
    literals only. Tokens that are already valid Python are left alone:
    all-zero literals (``00``), floats and fractional parts (``07.5``,
    ``1.07``), exponents (``1e07``, consumed as a name run), and ``0x``/
    ``0o``/``0b`` prefixes (the digit run stops at the prefix letter).
    """
    out: list[str] = []
    quote: str | None = None
    index, length = 0, len(text)
    while index < length:
        char = text[index]
        if quote is not None:
            out.append(char)
            if char == "\\" and index + 1 < length:
                out.append(text[index + 1])
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            out.append(char)
            index += 1
            continue
        if char.isalpha() or char == "_":
            end = index
            while end < length and (text[end].isalnum() or text[end] == "_"):
                end += 1
            out.append(text[index:end])
            index = end
            continue
        if char.isdigit():
            end = index
            while end < length and (text[end].isdigit() or text[end] == "_"):
                end += 1
            token = text[index:end]
            digits = token.replace("_", "")
            follower = text[end] if end < length else ""
            preceded_by_dot = index > 0 and text[index - 1] == "."
            if (
                digits[0] == "0"
                and digits.strip("0")
                and not preceded_by_dot
                and follower not in {".", "e", "E", "j", "J"}
            ):
                out.append(str(int(digits)))
            else:
                out.append(token)
            index = end
            continue
        out.append(char)
        index += 1
    return "".join(out)


_QUOTE_FOLLOWERS = {",", ")", "]", "}", ":"}


def escape_nested_quotes_in_strings(text: str) -> tuple[str, bool]:
    """Close a broken string literal at the only closing quote that works.

    Models emitting shell commands frequently nest unescaped same-style
    quotes inside a string argument — ``command='sed -n '360,450p' f.py'``,
    or a quoted ``python3 -c`` payload that itself contains quoted strings —
    which Python reads as juxtaposed garbage, so the call is dropped even
    though the intent is unambiguous. A string is treated as broken when
    its first unescaped quote cannot syntactically close it (what follows
    is none of ``,``, ``)``, ``]``, ``}``, ``:``). For a broken string,
    every syntactically plausible closing quote is tried: interior quotes
    escaped, the rest of the text kept verbatim, and the result (with
    control chars escaped) validated with ``ast.parse``. Exactly one
    candidate parsing means recovery — the decoded value is exactly the
    text the model wrote. Zero or several parsing candidates means the
    nesting is genuinely ambiguous and the text is returned unchanged
    rather than guessed at.

    Returns (rewritten_text, changed); run the result through
    escape_ctrl_chars_in_strings before parsing — quotes chosen here can
    move raw control chars inside the string.
    """

    def unescaped_quotes(start: int, quote: str) -> list[int]:
        positions = []
        j = start
        while j < len(text):
            if text[j] == "\\":
                j += 2
                continue
            if text[j] == quote:
                positions.append(j)
            j += 1
        return positions

    def is_closer(pos: int) -> bool:
        k = pos + 1
        while k < len(text) and text[k].isspace():
            k += 1
        return k < len(text) and text[k] in _QUOTE_FOLLOWERS

    prefix: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char not in {"'", '"'}:
            prefix.append(char)
            index += 1
            continue
        quotes = unescaped_quotes(index + 1, char)
        if not quotes:
            return text, False
        if is_closer(quotes[0]):
            # The normal reading closes this string; move past it.
            prefix.append(text[index : quotes[0] + 1])
            index = quotes[0] + 1
            continue
        winners = []
        for close in (j for j in quotes if is_closer(j)):
            interior: list[str] = []
            for j in range(index + 1, close):
                if text[j] == char and not _is_escaped(text, j):
                    interior.append("\\")
                interior.append(text[j])
            candidate = "".join(
                ["".join(prefix), char, "".join(interior), char, text[close + 1 :]]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                try:
                    ast.parse(escape_ctrl_chars_in_strings(candidate))
                except (SyntaxError, ValueError):
                    continue
            winners.append(candidate)
        if len(winners) == 1:
            return winners[0], True
        return text, False
    return text, False


def _is_escaped(text: str, index: int) -> bool:
    """Whether the character at ``index`` is backslash-escaped.

    A character is escaped iff it is preceded by an *odd* number of consecutive
    backslashes. Checking only the single preceding character is wrong for even
    runs: in ``'ab\\'`` the closing quote follows an escaped backslash (``\\\\``)
    and is therefore NOT escaped — it closes the string. Common in regex/code
    arguments such as ``r'\\\\b'``.
    """
    backslashes = 0
    j = index - 1
    while j >= 0 and text[j] == "\\":
        backslashes += 1
        j -= 1
    return backslashes % 2 == 1


def repair_candidates(text: str) -> list[str]:
    """Return rewrites of ``text`` to try in order, as vLLM's lfm2 parser does.

    Each rewrite is a no-op on already-valid text; the first one that parses
    wins. Call :func:`restore_reserved_kwarg_names` on the decoded arguments.
    """

    escaped = escape_ctrl_chars_in_strings(normalize_leading_zero_ints(text))
    candidates = [escaped]
    requoted, requote_changed = escape_nested_quotes_in_strings(escaped)
    if requote_changed:
        # Requoting can move raw control chars inside the string; escape again.
        candidates.append(escape_ctrl_chars_in_strings(requoted))
    renamed, kw_renamed = rename_reserved_kwargs(candidates[-1])
    if kw_renamed:
        candidates.append(renamed)
    return candidates

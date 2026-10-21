# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The IFEval constraint checks, keyed by instruction id such as ``keywords:letter_frequency``.

Each check takes the candidate text and the constraint's params and returns ``(passed, detail)``.
The detail is a short human-readable string that lands in the reward's ``detail`` field. Checks are
pure stdlib: this module has to run inside every task image without an extra.

``language:response_language`` describes an instruction the model was given rather than a property
that can be measured exactly, so its check is an approximation, and its docstring says so.

A word is a run of letters or digits in any script, with apostrophes and hyphens inside it, so
``don't`` and ``high-risk`` are one word each and a run of Han characters is one word.
"""

import json
import re
from collections import Counter
from collections.abc import Callable

Check = Callable[[str, dict], tuple[bool, str]]

CONSTRAINTS: dict[str, Check] = {}

PARAGRAPH_SEPARATOR = re.compile(r"\n\s*\n+")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
APOSTROPHES = "'\u2019"
WORD = re.compile(r"[^\W_](?:[^\W_]|['\u2019\-](?=[^\W_]))*")
BULLET = re.compile(r"^\s*[\*\-]\s+\S")
BULLET_ONLY_LINE = re.compile(r"^[\s*\-+]*$")
SQUARE_BRACKETS = re.compile(r"\[[^\[\]\n]*[^\[\]\s][^\[\]\n]*\]")
ONE_WORD_BRACKETS = re.compile(r"\[[^\[\]\s]*\]")
TITLE_MARKER = re.compile(r"<<[^<>\n]*[^<>\s][^<>\n]*>>")
DOUBLE_ANGLE_SPAN = re.compile(r"<<((?:(?!<<|>>)[^\n])*)>>")
JSON_FENCE = re.compile(r"^```(?:json)?\s*\n(.*)\n```\s*$", re.DOTALL)
MARKDOWN_HIGHLIGHT = re.compile(r"\*{1,2}[^*\n]+\*{1,2}")
# A sentence end followed by whitespace, or by a hyphen and then whitespace: sentences not joined by a hyphen.
UNJOINED_SENTENCE_END = re.compile(r"(\S*)([.!?])[\"')\]]*(?:\s|-\s)")
DIVIDER_SPLIT = re.compile(r"\s?\*\*\*\s?")
# The counting_composition prompt writes the divider as "* * *", the same Markdown divider as "***".
COMPOSITION_DIVIDER_SPLIT = re.compile(r"\s?(?:\*\*\*|\* \* \*)\s?")

CONSTRAINED_RESPONSES = ("My answer is yes.", "My answer is no.", "My answer is maybe.")
# Exactly six asterisks; a longer run is not the separator.
TWO_RESPONSE_SEPARATOR = re.compile(r"(?<!\*)\*{6}(?!\*)")
# Full stop, fullwidth and small full stops, one- and two-dot leaders, and the ellipsis "…".
DOTS = ".\uff0e\ufe52\u2024\u2025\u2026"
# Markdown emphasis or quotation marks around the whole answer, as in "**My answer is yes.**".
ANSWER_WRAPPING = re.compile(r'^(\*{1,2}|")(.*)\1$', re.DOTALL)
ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "cf", "no"})
SENTENCE_FINAL = ".!?"

# Closed-class words that are cheap evidence a response is in the requested language. Scripts with
# no entry (zh, ja, ko, ar, th) are checked by their share of non-ASCII characters instead.
LANGUAGE_HINTS: dict[str, set[str]] = {
    "en": {"the", "and", "of", "to", "is", "in", "a"},
    "es": {"el", "la", "y", "de", "que", "en", "un", "es"},
    "fr": {"le", "la", "et", "de", "que", "un", "une", "est"},
    "de": {"der", "die", "und", "ist", "ein", "eine", "zu", "von"},
    "it": {"il", "la", "e", "di", "che", "un", "una", "è"},
    "pt": {"o", "a", "e", "de", "que", "um", "uma", "é"},
    "ru": {"и", "в", "не", "на", "с", "но"},  # noqa: RUF001 - Cyrillic es, not a latin c.
    "zh": set(),
    "ja": set(),
    "ko": set(),
    "ar": set(),
    "hi": {"और", "है", "का", "की"},
    "th": set(),
    "vi": {"và", "là", "có", "không", "trong"},
    "tr": {"ve", "bir", "bu", "için", "ile"},
    "nl": {"de", "het", "en", "een", "van", "is"},
}
NON_ASCII_MIN = 10
NON_ASCII_SHARE = 0.2
PALINDROME_MIN_LETTERS = 3


def constraint(name: str) -> Callable[[Check], Check]:
    def register(check: Check) -> Check:
        CONSTRAINTS[name] = check
        return check

    return register


def paragraphs(text: str) -> list[str]:
    return [p for p in PARAGRAPH_SEPARATOR.split(text.strip()) if p.strip()]


def divided_parts(text: str, divider: re.Pattern[str]) -> list[str] | None:
    """Paragraphs separated by a markdown divider, or None when two dividers leave an empty one."""
    parts = divider.split(text.strip())
    if parts and not parts[0].strip():
        parts = parts[1:]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return None if any(not part.strip() for part in parts) else parts


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]


def words(text: str) -> list[str]:
    return WORD.findall(text)


def word_key(word: str) -> str:
    """The word lowercased, without a trailing possessive, so "Space's" and "space" share a key."""
    lowered = word.lower()
    return lowered[:-2] if len(lowered) > 2 and lowered[-2] in APOSTROPHES and lowered[-1] == "s" else lowered


def same_word(word: str, target: str) -> bool:
    return word_key(word) == word_key(target)


def count_keyword(text: str, keyword: str) -> int:
    return len(re.findall(r"\b" + re.escape(keyword) + r"\b", text, flags=re.IGNORECASE))


def contains_keyword(text: str, keyword: str, word_boundary: bool = True) -> bool:
    return count_keyword(text, keyword) > 0 if word_boundary else keyword.lower() in text.lower()


def compare(actual: int, relation: str, expected: int) -> tuple[bool, str] | None:
    """Apply an IFEval ``relation``, or ``None`` when the relation is not one IFEval defines."""
    detail = f"actual={actual} rel={relation} n={expected}"
    if relation == "at least":
        return actual >= expected, detail
    if relation == "less than":
        return actual < expected, detail
    if relation == "at most":
        return actual <= expected, detail
    if relation in ("equal", "exactly"):
        return actual == expected, detail
    return None


def _with_relation(actual: int, relation: str, expected: int, prefix: str) -> tuple[bool, str]:
    result = compare(actual, relation, expected)
    if result is None:
        return False, f"unsupported relation: {relation}"
    passed, detail = result
    return passed, f"{prefix} {detail}"


# === length_constraints =====================================================


@constraint("length_constraints:number_paragraphs")
def number_paragraphs(text: str, params: dict) -> tuple[bool, str]:
    """The prompt separates paragraphs with the markdown divider ``***``."""
    expected = params.get("num_paragraphs")
    if not isinstance(expected, int):
        return False, "missing num_paragraphs"
    parts = divided_parts(text, DIVIDER_SPLIT)
    if parts is None:
        return False, "empty paragraph between dividers"
    return len(parts) == expected, f"paragraphs={len(parts)} expected={expected}"


@constraint("length_constraints:number_words")
def number_words(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("num_words")
    if not isinstance(expected, int):
        return False, "missing num_words"
    return _with_relation(len(words(text)), params.get("relation", "at least"), expected, "words:")


@constraint("length_constraints:number_sentences")
def number_sentences(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("num_sentences")
    if not isinstance(expected, int):
        return False, "missing num_sentences"
    return _with_relation(len(sentences(text)), params.get("relation", "at least"), expected, "sentences:")


@constraint("length_constraints:nth_paragraph_first_word")
def nth_paragraph_first_word(text: str, params: dict) -> tuple[bool, str]:
    expected_paragraphs = params.get("num_paragraphs")
    nth = params.get("nth_paragraph")
    first_word = params.get("first_word")
    if not isinstance(expected_paragraphs, int) or not isinstance(nth, int) or not isinstance(first_word, str):
        return False, "missing kwargs"
    found = paragraphs(text)
    if len(found) != expected_paragraphs or nth < 1 or nth > len(found):
        return False, f"paragraphs={len(found)} need={expected_paragraphs} nth={nth}"
    actual = words(found[nth - 1])
    if not actual or not same_word(actual[0], first_word):
        return False, f"got first_word={actual[:1]} expected={first_word!r}"
    return True, "ok"


# === keywords ===============================================================


@constraint("keywords:forbidden_words")
def forbidden_words(text: str, params: dict) -> tuple[bool, str]:
    forbidden = params.get("forbidden_words", [])
    if not isinstance(forbidden, list) or not all(isinstance(word, str) for word in forbidden):
        return False, "forbidden_words must be a list of strings"
    if type(params.get("word_boundary", True)) is not bool:
        raise ValueError("word_boundary must be boolean")
    hits = [w for w in forbidden if contains_keyword(text, w, params.get("word_boundary", True))]
    return not hits, f"hits={hits}"


@constraint("keywords:word_once")
def word_once(text: str, params: dict) -> tuple[bool, str]:
    keyword = params.get("keyword")
    if not isinstance(keyword, str):
        return False, "missing keyword"
    count = count_keyword(text, keyword)
    return count == 1, f"count={count} keyword={keyword!r}"


@constraint("keywords:letter_frequency")
def letter_frequency(text: str, params: dict) -> tuple[bool, str]:
    letter = params.get("letter")
    expected = params.get("let_frequency")
    if not (isinstance(letter, str) and len(letter) == 1 and isinstance(expected, int)):
        return False, "bad kwargs"
    actual = sum(1 for c in text.lower() if c == letter.lower())
    return _with_relation(actual, params.get("let_relation", "at least"), expected, f"letter={letter}")


@constraint("keywords:existence")
def keyword_existence(text: str, params: dict) -> tuple[bool, str]:
    keywords = params.get("keywords", [])
    if not isinstance(keywords, list) or not all(isinstance(word, str) for word in keywords):
        return False, "keywords must be a list of strings"
    if type(params.get("word_boundary", True)) is not bool:
        raise ValueError("word_boundary must be boolean")
    missing = [w for w in keywords if not contains_keyword(text, w, params.get("word_boundary", True))]
    return not missing, f"missing={missing}"


@constraint("keywords:frequency")
def keyword_frequency(text: str, params: dict) -> tuple[bool, str]:
    keyword = params.get("keyword")
    expected = params.get("frequency")
    if not isinstance(keyword, str) or not isinstance(expected, int):
        return False, "bad kwargs"
    actual = count_keyword(text, keyword)
    return _with_relation(actual, params.get("relation", "at least"), expected, f"keyword={keyword!r}")


@constraint("keywords:word_count_different_numbers")
def word_count_different_numbers(text: str, params: dict) -> tuple[bool, str]:
    """Same params as ``keywords:frequency`` in the Nemotron dataset."""
    return keyword_frequency(text, params)


@constraint("keywords:keyword_specific_position")
def keyword_specific_position(text: str, params: dict) -> tuple[bool, str]:
    """The keyword must be the ``m``-th word of the ``n``-th sentence."""
    keyword = params.get("keyword")
    sentence_index = params.get("n")
    word_index = params.get("m")
    if not isinstance(keyword, str) or not isinstance(sentence_index, int) or not isinstance(word_index, int):
        return False, "bad kwargs"
    found = sentences(text)
    if sentence_index < 1 or sentence_index > len(found):
        return False, f"only {len(found)} sentences, need sentence {sentence_index}"
    in_sentence = words(found[sentence_index - 1])
    if word_index < 1 or word_index > len(in_sentence):
        return False, f"sentence has {len(in_sentence)} words, need pos {word_index}"
    actual = in_sentence[word_index - 1]
    return (
        same_word(actual, keyword),
        f"pos[{sentence_index},{word_index}]={actual!r} expected={keyword!r}",
    )


@constraint("keywords:palindrome")
def palindrome(text: str, params: dict) -> tuple[bool, str]:
    """The response must contain at least one palindromic word of three letters or more."""
    for word in words(text):
        lowered = word.lower()
        if len(lowered) >= PALINDROME_MIN_LETTERS and lowered == lowered[::-1]:
            return True, f"palindrome={lowered!r}"
    return False, "no palindrome found"


@constraint("keywords:no_adjacent_consecutive")
def no_adjacent_consecutive(text: str, params: dict) -> tuple[bool, str]:
    """No word may follow a word whose first letter comes just before its own, as in "apple banana".

    Like NVIDIA's checker, letters of any script are compared by code point, so "в городе" fails too.
    """
    initials = [w[0].lower() for w in words(text)]
    for index in range(1, len(initials)):
        left, right = initials[index - 1], initials[index]
        if left.isalpha() and right.isalpha() and ord(right) - ord(left) == 1:
            return False, f"adjacent words at pos {index} start with {left!r} and {right!r}"
    return True, "no adjacent consecutive initials"


@constraint("keywords:start_end")
def start_end(text: str, params: dict) -> tuple[bool, str]:
    """The response must start and end with the same word, with nothing after the last word."""
    stripped = text.strip()
    found = words(stripped)
    if not found:
        return False, "empty"
    if not stripped[-1].isalnum():
        return False, f"text ends with {stripped[-1]!r} after the last word"
    return same_word(found[0], found[-1]), f"start={found[0]!r} end={found[-1]!r}"


# === change_case ============================================================


@constraint("change_case:english_capital")
def english_capital(text: str, params: dict) -> tuple[bool, str]:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False, "no letters"
    passed = all(c.isupper() for c in letters)
    return passed, f"all_upper={passed}"


@constraint("change_case:english_lowercase")
def english_lowercase(text: str, params: dict) -> tuple[bool, str]:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False, "no letters"
    passed = all(c.islower() for c in letters)
    return passed, f"all_lower={passed}"


@constraint("change_case:capital_word_frequency")
def capital_word_frequency(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("capital_frequency")
    if not isinstance(expected, int):
        return False, "missing capital_frequency"
    actual = sum(1 for w in words(text) if len(w) >= 2 and w.isupper())
    return _with_relation(actual, params.get("capital_relation", "at least"), expected, "all_caps_words:")


# === punctuation ============================================================


@constraint("punctuation:no_comma")
def no_comma(text: str, params: dict) -> tuple[bool, str]:
    return "," not in text, f"has_comma={',' in text}"


@constraint("punctuation:punctuation_dot")
def no_dot(text: str, params: dict) -> tuple[bool, str]:
    """IFEval semantics: the response must contain no dots at all, an ellipsis "…" included."""
    found = sorted({c for c in text if c in DOTS})
    return not found, f"dots={found}"


@constraint("punctuation:punctuation_exclamation")
def no_exclamation(text: str, params: dict) -> tuple[bool, str]:
    return "!" not in text, f"has_exclam={'!' in text}"


# === startend ===============================================================


@constraint("startend:end_checker")
def end_checker(text: str, params: dict) -> tuple[bool, str]:
    end_phrase = params.get("end_phrase")
    if not isinstance(end_phrase, str):
        return False, "missing end_phrase"
    return text.rstrip().endswith(end_phrase), f"ends_with={end_phrase!r}"


@constraint("startend:quotation")
def quotation(text: str, params: dict) -> tuple[bool, str]:
    """The whole response must be wrapped in double quotation marks, straight or typographic."""
    stripped = text.strip()
    passed = len(stripped) > 1 and (
        (stripped.startswith('"') and stripped.endswith('"')) or (stripped.startswith("“") and stripped.endswith("”"))
    )
    return passed, f"quoted={passed}"


# === detectable_format ======================================================


@constraint("detectable_format:number_bullet_lists")
def number_bullet_lists(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("num_bullets")
    if not isinstance(expected, int):
        return False, "missing num_bullets"
    # A line of only markers, such as the divider "* * *" or "---", is not a bullet point.
    actual = sum(1 for line in text.splitlines() if BULLET.match(line) and not BULLET_ONLY_LINE.match(line))
    return actual == expected, f"bullets={actual} expected={expected}"


@constraint("detectable_format:title")
def title(text: str, params: dict) -> tuple[bool, str]:
    found = bool(TITLE_MARKER.search(text))
    return found, f"has_title_marker={found}"


@constraint("detectable_format:json_format")
def json_format(text: str, params: dict) -> tuple[bool, str]:
    stripped = text.strip()
    fence = JSON_FENCE.match(stripped)
    if fence:
        stripped = fence.group(1).strip()
    try:
        json.loads(stripped)
    except ValueError as error:
        return False, f"json_parse_error: {error}"
    return True, "json_parses"


@constraint("detectable_format:multiple_sections")
def multiple_sections(text: str, params: dict) -> tuple[bool, str]:
    """A section begins with a line that starts with the splitter and its number, after optional Markdown."""
    expected = params.get("num_sections")
    splitter = params.get("section_spliter") or params.get("section_splitter") or "Section"
    if not isinstance(expected, int) or not isinstance(splitter, str):
        return False, "bad kwargs"
    marker = re.compile(r"^[ \t#*_>]*" + re.escape(splitter) + r"\s*\d+", flags=re.IGNORECASE | re.MULTILINE)
    markers = list(marker.finditer(text))
    ends = [found.start() for found in markers[1:]] + [len(text)]
    bodies = [text[found.end() : end] for found, end in zip(markers, ends, strict=True)]
    if any(not WORD.search(body) for body in bodies):
        return False, "a section has no text after its marker"
    return len(markers) == expected, f"sections={len(markers)} expected={expected} splitter={splitter!r}"


@constraint("detectable_format:number_highlighted_sections")
def number_highlighted_sections(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("num_highlights")
    if not isinstance(expected, int):
        return False, "missing num_highlights"
    actual = len(MARKDOWN_HIGHLIGHT.findall(text))
    return actual >= expected, f"highlights={actual} need>={expected}"


def outside_spans(text: str, spans: list[re.Match]) -> str | None:
    """The first word of ``text`` that no span covers."""
    for word in WORD.finditer(text):
        if not any(span.start() < word.start() and word.end() < span.end() for span in spans):
            return word.group()
    return None


@constraint("detectable_format:square_brackets")
def square_brackets(text: str, params: dict) -> tuple[bool, str]:
    """Every word must sit in its own bracket span; punctuation may stay outside, as in "[Hello], [world]"."""
    if not WORD.search(text):
        return False, "no words"
    uncovered = outside_spans(text, list(ONE_WORD_BRACKETS.finditer(text)))
    return uncovered is None, f"word outside one-word brackets: {uncovered!r}" if uncovered else "every word bracketed"


@constraint("detectable_format:sentence_hyphens")
def sentence_hyphens(text: str, params: dict) -> tuple[bool, str]:
    """Sentences must be joined by hyphens with no whitespace between them, as in "One.-Two."."""
    stripped = text.strip()
    if not WORD.search(stripped):
        return False, "no sentences"
    for match in UNJOINED_SENTENCE_END.finditer(stripped):
        # The text before the dot back to the last hyphen, so "One.-e.g." is read as "e.g." and "One.-Two." as "Two.".
        token = match.group(1).rsplit("-", 1)[-1].lower()
        if match.group(2) == "." and ("." in token or token in ABBREVIATIONS):
            continue
        return False, f"sentence end without a hyphen after {match.group(0).strip()!r}"
    return True, "sentences joined by hyphens"


@constraint("detectable_format:bigram_wrapping")
def bigram_wrapping(text: str, params: dict) -> tuple[bool, str]:
    """Every word must sit in a <<...>> span of two words; the last span of a line or a sentence may hold one."""
    if not WORD.search(text):
        return False, "no words"
    for line in text.splitlines():
        spans = list(DOUBLE_ANGLE_SPAN.finditer(line))
        uncovered = outside_spans(line, spans)
        if uncovered is not None:
            return False, f"word outside <<...>>: {uncovered!r}"
        for index, span in enumerate(spans):
            count = len(words(span.group(1)))
            after = line[span.end() : span.end() + 1]
            closes = span.group(1).rstrip()[-1:] in SENTENCE_FINAL or after in SENTENCE_FINAL or index == len(spans) - 1
            if count != 2 and not (count == 1 and closes):
                return False, f"span {span.group(0)!r} holds {count} words"
    return True, "every bigram wrapped"


@constraint("detectable_format:constrained_response")
def constrained_response(text: str, params: dict) -> tuple[bool, str]:
    stripped = text.strip()
    wrapped = ANSWER_WRAPPING.match(stripped)
    if wrapped:
        stripped = wrapped.group(2).strip()
    passed = stripped in CONSTRAINED_RESPONSES
    return passed, f"matched_constrained={passed}"


# === detectable_content =====================================================


@constraint("detectable_content:postscript")
def postscript(text: str, params: dict) -> tuple[bool, str]:
    marker = params.get("postscript_marker")
    if not isinstance(marker, str):
        return False, "missing postscript_marker"
    match = re.search(r"^\s*" + re.escape(marker), text, flags=re.MULTILINE | re.IGNORECASE)
    # The postscript needs some text after its marker.
    found = match is not None and WORD.search(text, match.end()) is not None
    return found, f"has_postscript={marker!r}: {found}"


@constraint("detectable_content:number_placeholders")
def number_placeholders(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("num_placeholders")
    if not isinstance(expected, int):
        return False, "missing num_placeholders"
    actual = len(SQUARE_BRACKETS.findall(text))
    return actual >= expected, f"placeholders={actual} need>={expected}"


# === paragraphs =============================================================


@constraint("paragraphs:paragraphs")
def divided_paragraphs(text: str, params: dict) -> tuple[bool, str]:
    """Exactly two paragraphs, separated by the markdown divider ``***``."""
    parts = divided_parts(text, DIVIDER_SPLIT)
    if parts is None:
        return False, "empty paragraph between dividers"
    return len(parts) == 2, f"*** divided paragraphs={len(parts)}"


@constraint("paragraphs:paragraphs2")
def divided_paragraphs2(text: str, params: dict) -> tuple[bool, str]:
    """Exactly two paragraphs, separated by two line breaks."""
    parts = text.strip().split("\n\n")
    if any(not part.strip() for part in parts):
        return False, "empty paragraph between blank lines"
    return len(parts) == 2, f"blank-line paragraphs={len(parts)}"


# === first_word / last_word =================================================


@constraint("first_word:first_word_answer")
def first_word_answer(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("first_word")
    if not isinstance(expected, str):
        return False, "missing first_word"
    found = words(text)
    actual = found[0] if found else ""
    return same_word(actual, expected), f"first={actual!r} expected={expected!r}"


@constraint("first_word:first_word_sent")
def first_word_sent(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("first_word")
    if not isinstance(expected, str):
        return False, "missing first_word"
    found = sentences(text)
    if not found:
        return False, "no sentences"
    for sentence in found:
        in_sentence = words(sentence)
        if not in_sentence or not same_word(in_sentence[0], expected):
            return False, f"sent starts with {in_sentence[:1]} need={expected!r}"
    return True, f"all {len(found)} sentences start with {expected!r}"


@constraint("last_word:last_word_answer")
def last_word_answer(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("last_word")
    if not isinstance(expected, str):
        return False, "missing last_word"
    found = words(text)
    actual = found[-1] if found else ""
    return same_word(actual, expected), f"last={actual!r} expected={expected!r}"


@constraint("last_word:last_word_sent")
def last_word_sent(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("last_word")
    if not isinstance(expected, str):
        return False, "missing last_word"
    found = sentences(text)
    if not found:
        return False, "no sentences"
    for sentence in found:
        in_sentence = words(sentence.rstrip(".!?\"' "))
        if not in_sentence or not same_word(in_sentence[-1], expected):
            return False, f"sent ends with {in_sentence[-1:]} need={expected!r}"
    return True, f"all {len(found)} sentences end with {expected!r}"


# === count ==================================================================


@constraint("count:count_unique")
def count_unique(text: str, params: dict) -> tuple[bool, str]:
    """No word may appear twice, ignoring case and a trailing possessive."""
    counts = Counter(word_key(w) for w in words(text))
    if not counts:
        return False, "empty"
    repeated = sorted(word for word, count in counts.items() if count > 1)
    return not repeated, f"unique={len(counts)} total={counts.total()} repeated={repeated[:5]}"


@constraint("count:count_increment_word")
def count_increment_word(text: str, params: dict) -> tuple[bool, str]:
    """The first keyword group must appear exactly one more time than the second."""
    first_group = params.get("keyword1")
    second_group = params.get("keyword2")
    if not isinstance(first_group, list) or not isinstance(second_group, list):
        return False, "bad kwargs"
    first = sum(count_keyword(text, w) for w in first_group if isinstance(w, str))
    second = sum(count_keyword(text, w) for w in second_group if isinstance(w, str))
    passed = first > 0 and first == second + 1
    return passed, f"kw1_count={first} kw2_count={second} (need kw1 == kw2+1)"


@constraint("count:lowercase_counting")
def lowercase_counting(text: str, params: dict) -> tuple[bool, str]:
    """Reads "all lowercase words should appear at most N times" as NVIDIA's checker does: at most N in total."""
    expected = params.get("N")
    if not isinstance(expected, int):
        return False, "missing N"
    actual = sum(1 for w in words(text) if w.islower())
    return actual <= expected, f"lower_words={actual} need<={expected}"


@constraint("count:counting_composition")
def counting_composition(text: str, params: dict) -> tuple[bool, str]:
    """Three paragraphs between ``* * *`` dividers, each of n_sent sentences of n_words words."""
    expected_sentences = params.get("n_sent")
    expected_words = params.get("n_words")
    if not isinstance(expected_sentences, int) or not isinstance(expected_words, int):
        return False, "bad kwargs"
    parts = divided_parts(text, COMPOSITION_DIVIDER_SPLIT)
    if parts is None:
        return False, "empty paragraph between dividers"
    if len(parts) != 3:
        return False, f"paragraphs={len(parts)} need=3"
    for number, part in enumerate(parts):
        found = sentences(part)
        if len(found) != expected_sentences:
            return False, f"paragraph {number} has {len(found)} sentences, need={expected_sentences}"
        counts = [len(words(sentence)) for sentence in found]
        if any(count != expected_words for count in counts):
            return False, f"paragraph {number} sentence word counts {counts}, need={expected_words}"
    return True, f"3 paragraphs of {expected_sentences} sentences of {expected_words} words"


# === letters ================================================================


@constraint("letters:letter_counting")
def letter_counting(text: str, params: dict) -> tuple[bool, str]:
    """The prompt counts letters ("Answer with less than N letters"); like NVIDIA's checker, only a to z."""
    expected = params.get("N")
    if not isinstance(expected, int):
        return False, "missing N"
    actual = sum(1 for c in text if c.isascii() and c.isalpha())
    return _with_relation(actual, params.get("relation", "at least"), expected, "letters:")


@constraint("letters:letter_counting2")
def letter_counting2(text: str, params: dict) -> tuple[bool, str]:
    """Same params as ``keywords:letter_frequency``."""
    return letter_frequency(text, params)


# === language ===============================================================


@constraint("language:response_language")
def response_language(text: str, params: dict) -> tuple[bool, str]:
    """Approximate: closed-class hint words, or a share of non-ASCII for non-Latin scripts."""
    language = params.get("language")
    if not isinstance(language, str):
        return False, "missing language"
    language = language.lower()
    hints = LANGUAGE_HINTS.get(language)
    if hints:
        matched = any(contains_keyword(text, hint) for hint in hints)
        return matched, f"lang={language} hint_match={matched}"
    non_ascii = sum(1 for c in text if ord(c) > 127)
    passed = non_ascii >= max(NON_ASCII_MIN, int(NON_ASCII_SHARE * max(len(text), 1)))
    return passed, f"lang={language} non_ascii={non_ascii}"


# === copy ===================================================================


@constraint("copy:repeat_phrase")
def repeat_phrase(text: str, params: dict) -> tuple[bool, str]:
    """The phrase must appear exactly small_n times, each time with one word other than the first and last replaced."""
    phrase = params.get("phrase")
    expected = params.get("small_n")
    if not isinstance(phrase, str) or not isinstance(expected, int):
        return False, "bad kwargs"
    target = [word_key(w) for w in words(phrase)]
    found = [word_key(w) for w in words(text)]
    changes = []
    index = 0
    while target and index + len(target) <= len(found):
        differing = [j for j, word in enumerate(target) if found[index + j] != word]
        if len(differing) <= 1:
            changes.append(differing)
            index += len(target)
        else:
            index += 1
    inner = range(1, len(target) - 1)
    passed = len(changes) == expected and all(len(change) == 1 and change[0] in inner for change in changes)
    return passed, f"copies={len(changes)} need={expected} changed_positions={changes}"


# === combination ============================================================


@constraint("combination:two_responses")
def two_responses(text: str, params: dict) -> tuple[bool, str]:
    sections = TWO_RESPONSE_SEPARATOR.split(text)
    if any(not WORD.search(section) for section in sections[1:-1]):
        return False, "empty section between responses"
    # A part with no words, such as the "." after a closing separator, is not a response.
    parts = [section.strip() for section in sections if WORD.search(section)]
    passed = len(parts) == 2 and parts[0] != parts[1]
    return passed, f"sections_by_******={len(parts)}"

"""
Deterministic text normalisation for HTML-bearing corpora (LEV-12).

SODD's `first_post` / `second_post` are HTML fragments containing prose and
code snippets. They have to be reduced to plain text before they can be
embedded, and — because the D2 dataset is distributed as identifiers only —
the *same* reduction has to happen again at rehydration time on someone else's
machine. Any difference between the two produces a different query string and
breaks the byte-identical round-trip the replication criterion rests on.

That is why this is `html.parser` from the standard library rather than
BeautifulSoup or lxml: both of those can change their output across versions,
which would silently diverge two rehydrations of the same dataset. Stdlib
parsing pins the behaviour to the Python version, which the run manifest
already records.

The rule, recorded in the manifest as `NORMALISATION_RULE`:

- Tags are dropped; their text content is kept.
- Code blocks (`<code>`, `<pre>`) keep their text content like any other
  element — a duplicate-question pair whose distinguishing content is the code
  snippet must not be reduced to identical prose.
- Block-level tags become a single space, so text either side of a `</p>` does
  not run together into one word.
- HTML entities are unescaped.
- Runs of whitespace (including newlines and tabs) collapse to one space, and
  the result is stripped.
"""

from html import unescape
from html.parser import HTMLParser
from typing import List

#: Human-readable statement of the rule below, recorded in the run manifest so
#: a replicator can tell which normalisation produced the distributed ids.
NORMALISATION_RULE = (
    "stdlib html.parser: tags dropped, text content kept (including inside "
    "<code>/<pre>), block-level tags emit one space, entities unescaped, "
    "whitespace runs collapsed to a single space, result stripped"
)

#: Tags whose boundary is a word boundary: dropping them outright would join
#: the text on either side. `br` is here too even though it is void.
_BLOCK_TAGS = frozenset(
    {
        "address", "article", "aside", "blockquote", "br", "dd", "div", "dl",
        "dt", "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2",
        "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p",
        "pre", "section", "table", "tbody", "td", "tfoot", "th", "thead", "tr",
        "ul",
    }
)


class _TextExtractor(HTMLParser):
    """Collects the text content of an HTML fragment, dropping the markup."""

    def __init__(self) -> None:
        # convert_charrefs=True makes the parser unescape entities for us and
        # hand them through handle_data.
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _BLOCK_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _BLOCK_TAGS:
            self.parts.append(" ")

    def handle_startendtag(self, tag: str, attrs) -> None:
        self.parts.append(" ")


def normalize_html(text: str) -> str:
    """
    Reduce an HTML fragment to normalised plain text per `NORMALISATION_RULE`.

    Malformed markup is tolerated rather than rejected: unclosed and stray tags
    still yield their text content, because the corpus is real-world user
    content and a parse failure on one post is not a reason to lose the pair.
    """
    if not text:
        return ""
    parser = _TextExtractor()
    parser.feed(text)
    parser.close()
    # `convert_charrefs` handles well-formed entities; unescape() also catches
    # ones that arrive inside attribute-free stray text the parser passed through.
    return " ".join(unescape("".join(parser.parts)).split())

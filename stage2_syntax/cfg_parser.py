"""
Stage 2 -- Syntactic Processing (hand-built parser)
====================================================
Topic: Syntactic Processing (part 2/3)

This module implements a CYK (Cocke-Younger-Kasami) chart parser BY HAND --
no `nltk.ChartParser`, no `nltk.parse` grammar-parsing utilities are used.
The DP chart-filling algorithm, the unary-rule closure, and the backpointer
tree reconstruction are all written from scratch below.

The grammar is a small hand-written CFG (kept intentionally tiny -- 12
structural rules + a lexicon) covering simple declarative sentences such as:

    "the dog chased the cat"
    "a student wrote the important report"
    "the teacher sent the email to the student"

Grammar (structural, non-lexical rules -- 12 rules)
----------------------------------------------------
 1. S    -> NP VP
 2. NP   -> Det N
 3. NP   -> Det AdjN
 4. AdjN -> Adj N
 5. NP   -> Adj N
 6. NP   -> PropN
 7. VP   -> V NP
 8. VP   -> V
 9. VP   -> VP PP
10. VP   -> V PP
11. PP   -> P NP
12. NP   -> NP PP

Everything else (Det -> 'the', N -> 'dog', ...) is lexical/terminal and
lives in `LEXICON` below, exactly like a normal CFG's terminal productions.
"""

from __future__ import annotations
from collections import defaultdict

# ---------------------------------------------------------------------------
# Grammar definition
# ---------------------------------------------------------------------------
# Binary rules: LHS -> (RHS1, RHS2)   (both non-terminals)
BINARY_RULES = [
    ("S", ("NP", "VP")),
    ("NP", ("Det", "N")),
    ("NP", ("Det", "AdjN")),
    ("AdjN", ("Adj", "N")),
    ("NP", ("Adj", "N")),
    ("VP", ("V", "NP")),
    ("VP", ("VP", "PP")),
    ("VP", ("V", "PP")),
    ("PP", ("P", "NP")),
    ("NP", ("NP", "PP")),
]

# Unary non-terminal rules: LHS -> RHS (single non-terminal). CYK needs a
# closure step over these so e.g. "VP -> V" chains correctly.
UNARY_RULES = [
    ("NP", "PropN"),
    ("VP", "V"),
]

LEXICON = {
    "the": ["Det"], "a": ["Det"], "an": ["Det"],
    "dog": ["N"], "cat": ["N"], "student": ["N"], "report": ["N"],
    "teacher": ["N"], "book": ["N"], "email": ["N"], "meeting": ["N"],
    "class": ["N"], "professor": ["N"], "assignment": ["N"],
    "deadline": ["N"], "homework": ["N"], "exam": ["N"], "grade": ["N"],
    "message": ["N"], "computer": ["N"], "printer": ["N"], "library": ["N"],
    "attachment": ["N"], "link": ["N"], "file": ["N"], "form": ["N"],
    "chased": ["V"], "wrote": ["V"], "read": ["V"], "saw": ["V"],
    "sent": ["V"], "reviewed": ["V"], "finished": ["V"], "submitted": ["V"],
    "lazy": ["Adj"], "quick": ["Adj"], "big": ["Adj"], "important": ["Adj"],
    "late": ["Adj"], "final": ["Adj"],
    "to": ["P"], "on": ["P"], "with": ["P"], "before": ["P"], "in": ["P"],
    "john": ["PropN"], "mary": ["PropN"], "sam": ["PropN"], "priya": ["PropN"],
}


class Node:
    """Minimal parse-tree node used for backpointer reconstruction."""
    __slots__ = ("label", "children", "word")

    def __init__(self, label, children=None, word=None):
        self.label = label
        self.children = children or []
        self.word = word

    def to_nltk_tree(self):
        """Convert to an nltk.Tree so we can reuse nltk's pretty-printer /
        drawing utilities for the *output* (the grammar + CYK algorithm
        itself is still entirely hand-written)."""
        from nltk import Tree
        if self.word is not None:
            return Tree(self.label, [self.word])
        return Tree(self.label, [c.to_nltk_tree() for c in self.children])

    def to_bracket_string(self):
        if self.word is not None:
            return f"({self.label} {self.word})"
        return f"({self.label} " + " ".join(c.to_bracket_string() for c in self.children) + ")"


def _unary_closure(cell_dict, unary_rules):
    """Given a dict {symbol: Node} for one chart cell, repeatedly apply
    unary rules (LHS -> RHS) until no new symbols can be derived. This is
    the standard CYK extension needed to support unit productions like
    VP -> V without first converting the grammar to strict CNF."""
    changed = True
    while changed:
        changed = False
        for lhs, rhs in unary_rules:
            if rhs in cell_dict and lhs not in cell_dict:
                cell_dict[lhs] = Node(lhs, children=[cell_dict[rhs]])
                changed = True


def cyk_parse(tokens: list[str]):
    """Hand-written CYK chart parser.

    chart[i][j] (0-indexed, span length j-i) holds a dict mapping
    non-terminal symbol -> a Node (one valid derivation is kept per symbol,
    which is enough to demonstrate parsing mechanics; a full parse forest
    would store all derivations).

    Returns the Node for 'S' spanning the whole sentence, or None if the
    tiny grammar cannot parse the sentence (it will not cover every English
    sentence -- that's expected/acceptable per the assignment brief).
    """
    n = len(tokens)
    if n == 0:
        return None

    # chart[length][start] = {symbol: Node}
    chart = [[dict() for _ in range(n)] for _ in range(n + 1)]

    # base case: spans of length 1 -- lexical lookups
    for i, tok in enumerate(tokens):
        cell = chart[1][i]
        for tag in LEXICON.get(tok.lower(), []):
            cell[tag] = Node(tag, word=tok)
        _unary_closure(cell, UNARY_RULES)

    # recursive case: spans of length 2..n, classic CYK triple loop
    for length in range(2, n + 1):
        for start in range(0, n - length + 1):
            cell = chart[length][start]
            for split in range(1, length):
                left = chart[split][start]
                right = chart[length - split][start + split]
                for lhs, (r1, r2) in BINARY_RULES:
                    if r1 in left and r2 in right:
                        if lhs not in cell:
                            cell[lhs] = Node(lhs, children=[left[r1], right[r2]])
            _unary_closure(cell, UNARY_RULES)

    top = chart[n][0]
    return top.get("S")


def parse_sentence(tokens: list[str]):
    """Convenience wrapper: returns (Node or None, success: bool)."""
    root = cyk_parse(tokens)
    return root, root is not None


if __name__ == "__main__":
    tests = [
        "the dog chased the cat",
        "a student wrote the important report",
        "the teacher sent the email to the student",
        "mary read the book",
    ]
    for s in tests:
        toks = s.split()
        root, ok = parse_sentence(toks)
        print(s, "->", "PARSED" if ok else "NO PARSE (outside tiny grammar's coverage)")
        if ok:
            print(" ", root.to_bracket_string())

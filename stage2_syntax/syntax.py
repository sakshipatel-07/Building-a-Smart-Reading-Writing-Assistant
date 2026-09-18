"""
Stage 2 -- Syntactic Processing
================================
Topic: Syntactic Processing

Two complementary components, as required by the assignment:

1. `spacy_parse_sentence` -- uses spaCy (a real library) to produce a
   dependency parse tree + POS tag sequence for an arbitrary sentence. This
   is the "use a parser to produce a dependency tree" requirement.

2. `cfg_parser.py` (imported here) -- a hand-written CYK chart parser for a
   small custom CFG, proving understanding of parsing *mechanics* rather
   than just calling a library. This is the "implement by hand" requirement.

Both outputs are produced per sentence and a simple bracketed / nltk.tree
rendering is generated for the visual parse tree deliverable.
"""

from __future__ import annotations
import spacy
from nltk import Tree

from stage2_syntax.cfg_parser import parse_sentence as cyk_parse_sentence

_NLP = None


def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def split_sentences(text: str):
    nlp = get_nlp()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def spacy_parse_sentence(sentence: str):
    """Return POS tags and a dependency-tree structure for one sentence
    using spaCy (dependency parsing + POS tagging)."""
    nlp = get_nlp()
    doc = nlp(sentence)

    pos_sequence = [{"text": t.text, "pos": t.pos_, "tag": t.tag_} for t in doc]

    dependencies = [
        {"text": t.text, "dep": t.dep_, "head": t.head.text, "head_i": t.head.i, "i": t.i}
        for t in doc
    ]

    # Build an nltk.Tree from the dependency structure so we can render it
    # visually (dependency tree, rooted at the sentence's syntactic head).
    def build_dep_tree(token):
        children = [build_dep_tree(child) for child in token.children]
        label = f"{token.text}_{token.dep_}"
        if children:
            return Tree(label, children)
        return Tree(label, [])

    roots = [t for t in doc if t.head == t]
    dep_tree_str = "\n".join(str(build_dep_tree(r)) for r in roots)

    return {
        "sentence": sentence,
        "pos_sequence": pos_sequence,
        "dependencies": dependencies,
        "dep_tree_pretty": dep_tree_str,
    }


def cfg_parse_sentence(sentence: str):
    """Try the hand-built CYK parser on the (lower-cased, punctuation
    stripped) sentence. Not every sentence will be covered by the tiny
    12-rule grammar -- that is expected."""
    tokens = [t.strip(".,!?;:").lower() for t in sentence.split()]
    tokens = [t for t in tokens if t]
    root, ok = cyk_parse_sentence(tokens)
    return {
        "tokens": tokens,
        "parsed": ok,
        "bracket_tree": root.to_bracket_string() if ok else None,
    }


def process_syntax(text: str):
    """Full Stage-2 entry point: split into sentences, run both parsers on
    each sentence, return a list of per-sentence results."""
    results = []
    for sent in split_sentences(text):
        spacy_result = spacy_parse_sentence(sent)
        cfg_result = cfg_parse_sentence(sent)
        results.append({
            "sentence": sent,
            "spacy": spacy_result,
            "cfg": cfg_result,
        })
    return results


if __name__ == "__main__":
    demo = "The dog chased the cat. A student wrote the important report before the deadline."
    for r in process_syntax(demo):
        print("SENTENCE:", r["sentence"])
        print(" POS:", [(p["text"], p["pos"]) for p in r["spacy"]["pos_sequence"]])
        print(" DEP TREE (spaCy):")
        print(r["spacy"]["dep_tree_pretty"])
        print(" CFG (hand-built CYK):", r["cfg"]["bracket_tree"] or "no parse")
        print()

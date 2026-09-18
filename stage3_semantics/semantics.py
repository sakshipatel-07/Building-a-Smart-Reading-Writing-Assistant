"""
Stage 3 -- Semantic Analysis
=============================
Topic: Semantic Analysis

Three components:

1. Named Entity Recognition -- via spaCy's NER pipeline.
2. Semantic roles (who did what to whom) -- a HAND-BUILT rule set that walks
   the spaCy dependency tree for each sentence (the "harder/more instructive
   route" from the brief: no AllenNLP SRL model, just dependency-label
   rules: nsubj -> agent, dobj/attr/dative -> patient, ROOT verb -> action,
   plus prepositional-phrase modifiers).
3. Word-sense disambiguation -- a Lesk-algorithm-style implementation
   written from scratch: for a target ambiguous word, compare the overlap
   between (a) the surrounding sentence's content words and (b) each
   WordNet sense's gloss + examples, and pick the sense with the highest
   overlap. WordNet itself (via NLTK) supplies the sense inventory / glosses
   -- exactly the "WordNet overlap" approach requested.

Output: for each sentence, a structured dict of the shape
    {"agent": ..., "action": ..., "patient": ..., "entities": [...], "wsd": [...]}
"""

from __future__ import annotations
import re
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as nltk_stopwords

try:
    STOPWORDS = set(nltk_stopwords.words("english"))
except LookupError:
    import nltk
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(nltk_stopwords.words("english"))

from stage2_syntax.syntax import get_nlp

# Words the assignment explicitly calls out as ambiguous test cases, plus a
# few more common ones. WSD is attempted for any of these that appear in the
# text (this comfortably covers the ">= 3 ambiguous words" requirement).
AMBIGUOUS_WORDS = {"bank", "bat", "light", "book", "spring", "match", "bark"}

POS_TO_WORDNET = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}


# ---------------------------------------------------------------------------
# 1. Named Entity Recognition (spaCy)
# ---------------------------------------------------------------------------
def extract_entities(doc):
    return [{"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents]


# ---------------------------------------------------------------------------
# 2. Hand-built semantic-role extraction over the dependency tree
# ---------------------------------------------------------------------------
def extract_semantic_frame(sent_doc):
    """Rule-based SRL: find the main verb (ROOT), then read off its
    dependency children to assign coarse semantic roles. This mirrors the
    classic "who did what to whom" frame using dependency-label heuristics
    rather than a trained SRL model."""
    root = None
    for tok in sent_doc:
        if tok.dep_ == "ROOT":
            root = tok
            break
    if root is None:
        return {"agent": None, "action": None, "patient": None, "modifiers": []}

    agent = None
    patient = None
    recipient = None
    modifiers = []

    for child in root.children:
        if child.dep_ in ("nsubj", "nsubjpass", "expl"):
            agent = " ".join(t.text for t in child.subtree)
        elif child.dep_ in ("dobj", "attr", "oprd"):
            patient = " ".join(t.text for t in child.subtree)
        elif child.dep_ == "dative":
            # e.g. "sent [the report] [to Sarah]" -- 'dative' here heads the
            # recipient phrase, distinct from the direct-object patient.
            recipient = " ".join(t.text for t in child.subtree)
        elif child.dep_ == "prep":
            modifiers.append(" ".join(t.text for t in child.subtree))
        elif child.dep_ in ("ccomp", "xcomp"):
            if patient is None:
                patient = " ".join(t.text for t in child.subtree)

    if recipient:
        modifiers.insert(0, recipient)

    action = root.lemma_
    # passive voice: swap semantics so "agent" reported is the logical
    # patient marked nsubjpass, and look for a "by X" agent phrase.
    if any(c.dep_ == "nsubjpass" for c in root.children):
        for child in root.children:
            if child.dep_ == "agent":  # "by ..."
                for gc in child.children:
                    if gc.dep_ == "pobj":
                        agent, patient = " ".join(t.text for t in gc.subtree), agent

    return {"agent": agent, "action": action, "patient": patient, "modifiers": modifiers}


# ---------------------------------------------------------------------------
# 3. Hand-built Lesk word-sense disambiguation
# ---------------------------------------------------------------------------
def _tokenize_words(s: str):
    return set(w.lower() for w in re.findall(r"[a-zA-Z]+", s) if w.lower() not in STOPWORDS)


def lesk_wsd(target_word: str, sentence_tokens: list[str], pos: str | None = None):
    """From-scratch Lesk algorithm:
        1. Build a 'signature' set of content words from the target's
           sentence context.
        2. For every WordNet sense of target_word, build a signature set
           from its gloss + usage examples.
        3. Score each sense by the size of the overlap between the two sets.
        4. Return the sense with the maximum overlap (ties -> WordNet's
           default/most-frequent sense as a sensible fallback).
    """
    context = set(w.lower() for w in sentence_tokens if w.lower() not in STOPWORDS)
    context.discard(target_word.lower())

    wn_pos = POS_TO_WORDNET.get(pos) if pos else None
    synsets = wn.synsets(target_word, pos=wn_pos) if wn_pos else wn.synsets(target_word)
    if not synsets:
        return None

    best_sense = synsets[0]
    best_score = -1
    scored = []
    for syn in synsets:
        gloss_text = syn.definition() + " " + " ".join(syn.examples())
        signature = _tokenize_words(gloss_text)
        overlap = len(signature & context)
        scored.append((syn.name(), overlap, syn.definition()))
        if overlap > best_score:
            best_score = overlap
            best_sense = syn

    return {
        "word": target_word,
        "chosen_sense": best_sense.name(),
        "definition": best_sense.definition(),
        "overlap_score": best_score,
        "all_senses_scored": scored,
    }


def disambiguate_sentence(sent_doc):
    """Run Lesk WSD on every token in the sentence whose lemma is in
    AMBIGUOUS_WORDS (covers the required >=3 ambiguous test words, and
    generalizes to any others that show up in real input)."""
    tokens_text = [t.text for t in sent_doc]
    results = []
    for tok in sent_doc:
        lemma = tok.lemma_.lower()
        if lemma in AMBIGUOUS_WORDS and tok.pos_ in POS_TO_WORDNET:
            res = lesk_wsd(lemma, tokens_text, pos=tok.pos_)
            if res:
                results.append(res)
    return results


# ---------------------------------------------------------------------------
# Full stage-3 entry point
# ---------------------------------------------------------------------------
def process_semantics(text: str):
    nlp = get_nlp()
    doc = nlp(text)
    results = []
    for sent in doc.sents:
        sent_doc = sent.as_doc()
        frame = extract_semantic_frame(sent_doc)
        entities = extract_entities(sent_doc)
        wsd = disambiguate_sentence(sent_doc)
        results.append({
            "sentence": sent.text.strip(),
            "agent": frame["agent"],
            "action": frame["action"],
            "patient": frame["patient"],
            "modifiers": frame["modifiers"],
            "entities": entities,
            "wsd": wsd,
        })
    return results


if __name__ == "__main__":
    demo = ("John sent the report to Sarah before the meeting. "
             "I sat on the river bank and watched the bat fly at dusk. "
             "The light in the office was too dim to read the book.")
    for r in process_semantics(demo):
        print("SENT:", r["sentence"])
        print(" frame:", {"agent": r["agent"], "action": r["action"], "patient": r["patient"]})
        print(" entities:", r["entities"])
        for w in r["wsd"]:
            print(" WSD:", w["word"], "->", w["chosen_sense"], "-", w["definition"])
        print()

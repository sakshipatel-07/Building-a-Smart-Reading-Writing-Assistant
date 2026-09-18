"""
Stage 4 -- Discourse & Pragmatic Processing
=============================================
Topic: Discourse and Pragmatic Processing

Three components:

1. Coreference resolution across sentences in a paragraph. A pretrained
   neural coref model (neuralcoref / spaCy's coref component) is not
   reliably installable offline for every environment, so this module
   implements the assignment's explicitly-offered fallback: a HAND-BUILT
   heuristic resolver. For every pronoun it walks backwards over previously
   mentioned entities/noun phrases and picks the nearest antecedent that
   agrees in gender and number (using small hand-built gender/number
   look-up tables plus spaCy's morphological features).

2. Discourse-connective detection: a small hand-built rule list of
   connectives (however, therefore, because, ...) mapped to the discourse
   relation they typically signal (contrast, cause, elaboration, ...).

3. One pragmatic-inference rule: detect indirect requests phrased as polite
   yes/no questions ("Could you send the file?") and label their actual
   illocutionary force as a request, via pattern matching over the parsed
   (POS/dependency) structure.
"""

from __future__ import annotations
import re
from stage2_syntax.syntax import get_nlp

# ---------------------------------------------------------------------------
# 1. Heuristic coreference resolution
# ---------------------------------------------------------------------------
MALE_PRONOUNS = {"he", "him", "his", "himself"}
FEMALE_PRONOUNS = {"she", "her", "hers", "herself"}
NEUTRAL_SG_PRONOUNS = {"it", "its", "itself"}
PLURAL_PRONOUNS = {"they", "them", "their", "theirs", "themselves"}
ALL_PRONOUNS = MALE_PRONOUNS | FEMALE_PRONOUNS | NEUTRAL_SG_PRONOUNS | PLURAL_PRONOUNS
# 1st/2nd person pronouns are never resolved (no antecedent needed) and must
# also never themselves be offered as a candidate antecedent for some other
# pronoun ("it" should never resolve to "you").
NON_ANTECEDENT_PRONOUNS = ALL_PRONOUNS | {"i", "you", "we", "me", "us", "our", "your", "my", "ours", "yours"}

# A tiny common-name gender lookup used only to help agreement matching when
# no pretrained coref/gender model is available (heuristic, not exhaustive).
COMMON_MALE_NAMES = {"john", "sam", "mike", "david", "james", "robert", "michael", "tom", "raj", "arjun"}
COMMON_FEMALE_NAMES = {"mary", "sarah", "priya", "emma", "anna", "lisa", "susan", "linda", "neha", "kavya"}


def _entity_gender_number(chunk_text: str, root_token):
    """Very small heuristic: guess {'male','female','neutral','plural',
    'unknown'} for a candidate antecedent noun phrase."""
    head = chunk_text.split()[-1].lower().strip(".,")
    if root_token.tag_ in ("NNS", "NNPS"):
        return "plural"
    if head in COMMON_MALE_NAMES:
        return "male"
    if head in COMMON_FEMALE_NAMES:
        return "female"
    if root_token.ent_type_ == "PERSON":
        return "unknown_person"  # person but gender not guessable -> matches he/she loosely
    return "neutral"


def _pronoun_class(pronoun: str):
    p = pronoun.lower()
    if p in MALE_PRONOUNS:
        return "male"
    if p in FEMALE_PRONOUNS:
        return "female"
    if p in PLURAL_PRONOUNS:
        return "plural"
    if p in NEUTRAL_SG_PRONOUNS:
        return "neutral"
    return None


def resolve_coreferences(text: str):
    """Hand-built nearest-antecedent heuristic resolver.

    Walks the document token by token; maintains a list of candidate
    antecedents (noun chunks seen so far, most recent first). For every
    pronoun, picks the nearest prior noun chunk whose guessed
    gender/number class is compatible.

    Returns a list of coref "chains": {"pronoun": ..., "position": ...,
    "antecedent": ..., "sentence_index": ...}.
    """
    nlp = get_nlp()
    doc = nlp(text)

    # collect noun chunks with their sentence index, in document order
    chunk_list = []
    for i, sent in enumerate(doc.sents):
        for chunk in sent.noun_chunks:
            chunk_list.append((chunk.start, chunk.text, chunk.root, i))

    resolved = []
    for i, sent in enumerate(doc.sents):
        for tok in sent:
            if tok.text.lower() in ALL_PRONOUNS:
                pcls = _pronoun_class(tok.text)
                # candidates: noun chunks strictly before this pronoun's token index
                candidates = [c for c in chunk_list if c[0] < tok.i]
                antecedent = None
                for start, ctext, croot, csent in reversed(candidates):
                    if ctext.lower().strip() in NON_ANTECEDENT_PRONOUNS:
                        continue  # don't resolve a pronoun to another pronoun
                    gclass = _entity_gender_number(ctext, croot)
                    compatible = (
                        gclass == pcls
                        or (gclass == "unknown_person" and pcls in ("male", "female"))
                        or (gclass == "neutral" and pcls == "neutral")
                    )
                    if compatible:
                        antecedent = ctext
                        break
                resolved.append({
                    "pronoun": tok.text,
                    "sentence_index": i,
                    "antecedent": antecedent,
                })
    return resolved


# ---------------------------------------------------------------------------
# 2. Discourse connectives -> relation type (hand-built rule list)
# ---------------------------------------------------------------------------
DISCOURSE_CONNECTIVES = {
    "however": "contrast", "but": "contrast", "although": "contrast",
    "though": "contrast", "yet": "contrast", "whereas": "contrast",
    "therefore": "cause_effect", "thus": "cause_effect", "hence": "cause_effect",
    "so": "cause_effect", "because": "cause_effect", "since": "cause_effect",
    "consequently": "cause_effect", "as a result": "cause_effect",
    "moreover": "elaboration", "furthermore": "elaboration", "additionally": "elaboration",
    "in addition": "elaboration", "also": "elaboration", "for example": "elaboration",
    "for instance": "elaboration",
    "meanwhile": "temporal", "then": "temporal", "afterwards": "temporal",
    "before": "temporal", "after": "temporal", "subsequently": "temporal",
    "if": "condition", "unless": "condition",
}


def detect_discourse_relations(text: str):
    """Scan for connective phrases (case-insensitive, whole-word match, and
    a couple of multi-word phrases) and label the discourse relation each
    signals."""
    found = []
    lowered = text.lower()
    for phrase, relation in DISCOURSE_CONNECTIVES.items():
        for m in re.finditer(r"\b" + re.escape(phrase) + r"\b", lowered):
            found.append({
                "connective": phrase,
                "relation": relation,
                "char_span": [m.start(), m.end()],
                "context": text[max(0, m.start() - 30): m.end() + 30].strip(),
            })
    found.sort(key=lambda f: f["char_span"][0])
    return found


# ---------------------------------------------------------------------------
# 3. Pragmatic inference: indirect speech acts / requests
# ---------------------------------------------------------------------------
INDIRECT_REQUEST_PATTERNS = [
    re.compile(r"^\s*could you\b.*\?\s*$", re.IGNORECASE),
    re.compile(r"^\s*can you\b.*\?\s*$", re.IGNORECASE),
    re.compile(r"^\s*would you\b.*\?\s*$", re.IGNORECASE),
    re.compile(r"^\s*would you mind\b.*\?\s*$", re.IGNORECASE),
    re.compile(r"^\s*is it possible (for you )?to\b.*\?\s*$", re.IGNORECASE),
]


def detect_pragmatic_intent(sentence: str):
    """Pattern-matching pragmatic-inference rule: a surface yes/no question
    of the form "Could/Can/Would you ...?" is, pragmatically, almost always
    an indirect REQUEST for action rather than a literal query about the
    listener's ability/willingness. We detect the pattern and rewrite the
    literal question into the implied directive.
    """
    stripped = sentence.strip()
    for pat in INDIRECT_REQUEST_PATTERNS:
        if pat.match(stripped):
            # turn "Could you send the file?" into "send the file"
            body = re.sub(
                r"^(could|can|would) you( mind)?\s*",
                "", stripped, flags=re.IGNORECASE
            )
            body = re.sub(r"^is it possible (for you )?to\s*", "", body, flags=re.IGNORECASE)
            body = body.rstrip("?").strip()
            if body.endswith("ing"):  # "mind sending the file" -> keep gerund form as-is
                implied = body
            else:
                implied = body
            return {
                "surface_form": "yes_no_question",
                "literal_reading": stripped,
                "pragmatic_act": "indirect_request",
                "implied_directive": implied[0].upper() + implied[1:] if implied else implied,
            }
    if stripped.endswith("?"):
        return {"surface_form": "question", "literal_reading": stripped,
                "pragmatic_act": "literal_question", "implied_directive": None}
    return {"surface_form": "statement", "literal_reading": stripped,
            "pragmatic_act": "assertion", "implied_directive": None}


# ---------------------------------------------------------------------------
# Full stage-4 entry point
# ---------------------------------------------------------------------------
def process_discourse(text: str):
    nlp = get_nlp()
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents]

    coref_chains = resolve_coreferences(text)
    discourse_relations = detect_discourse_relations(text)
    pragmatics = [detect_pragmatic_intent(s) for s in sentences]

    return {
        "coref_chains": coref_chains,
        "discourse_relations": discourse_relations,
        "pragmatics": pragmatics,
    }


if __name__ == "__main__":
    demo = ("Sarah finished the report. However, she forgot to attach it. "
             "Because the deadline was close, she sent it late. "
             "Could you review it before the meeting?")
    result = process_discourse(demo)
    print("COREF CHAINS:")
    for c in result["coref_chains"]:
        print(" ", c)
    print("DISCOURSE RELATIONS:")
    for d in result["discourse_relations"]:
        print(" ", d["connective"], "->", d["relation"])
    print("PRAGMATICS:")
    for p in result["pragmatics"]:
        print(" ", p)

"""
Stage 1 -- Spell Checking
=========================
Topic: Spell Checking

What this module demonstrates
------------------------------
1. A from-scratch implementation of the Levenshtein edit-distance algorithm
   (no library calls such as `editdistance`, `python-Levenshtein`, or
   `difflib.get_close_matches`) used to find correction candidates against a
   word-frequency dictionary built from the NLTK Brown corpus.
2. A context-aware layer on top of the raw edit-distance ranking: when two or
   more candidates are equally close (or when a word is a real word but is a
   commonly *confused* word -- "there" / "their" / "they're" etc.), the
   surrounding bigram frequency (an n-gram language-model probability) is
   used to choose the more likely correction instead of just the closest
   string.
3. A diff/explanation output that records, for every change, *why* it was
   made (raw edit-distance score vs. n-gram probability boost).

Author note: everything below (Levenshtein DP table, candidate generation,
scoring, tokenisation) is hand written. The only external data used is a
frequency table built once from the Brown corpus (data/word_freq.json /
data/bigrams.json), which plays the role of "the dictionary / word-frequency
list" the assignment asks for.
"""

from __future__ import annotations
import json
import re
import string
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# 1. Levenshtein edit distance -- implemented from scratch (classic DP)
# ---------------------------------------------------------------------------
def levenshtein(a: str, b: str) -> int:
    """Damerau-Levenshtein edit distance (restricted / "optimal string
    alignment" variant), implemented from scratch as a full DP table.

    dp[i][j] = edit distance between a[:i] and b[:j] allowing four
    operations, each cost 1: insertion, deletion, substitution, and the
    transposition of two *adjacent* characters (a[i-1] <-> a[i-2]). The
    transposition rule is what plain Levenshtein lacks, and it is exactly
    what turns common typos like "recieve" -> "receive" (an adjacent
    ei/ie swap) into a distance-1 edit instead of distance-2, so the
    dictionary lookup ranks the right correction first.
    """
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    # full table needed here (not the rolling 1-row version) because the
    # transposition case looks two rows back.
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
            if (i > 1 and j > 1
                    and a[i - 1] == b[j - 2]
                    and a[i - 2] == b[j - 1]):
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)  # transposition
    return dp[n][m]


def bounded_levenshtein(a: str, b: str, max_dist: int = 2) -> int | None:
    """Same DP as `levenshtein` but returns None early if the distance is
    provably going to exceed max_dist. Used to prune the dictionary scan so
    spell-checking stays fast on a ~40k word dictionary."""
    if abs(len(a) - len(b)) > max_dist:
        return None
    d = levenshtein(a, b)
    return d if d <= max_dist else None


# ---------------------------------------------------------------------------
# 2. Dictionary + n-gram language model (built once from the Brown corpus)
# ---------------------------------------------------------------------------
class LanguageModel:
    def __init__(self):
        with open(DATA_DIR / "word_freq.json") as f:
            self.freq: dict[str, int] = json.load(f)
        self.total = sum(self.freq.values())
        self.vocab = set(self.freq.keys())

        bigram_path = DATA_DIR / "bigrams.json"
        if bigram_path.exists():
            with open(bigram_path) as f:
                self.bigrams: dict[str, dict[str, int]] = json.load(f)
        else:
            self.bigrams = {}

    def unigram_prob(self, word: str) -> float:
        return (self.freq.get(word, 0) + 1) / (self.total + len(self.vocab))

    def bigram_prob(self, prev_word: str, word: str) -> float:
        """P(word | prev_word) with add-1 smoothing, backing off to the
        unigram probability of `word` when we have no bigram statistics for
        prev_word. This is the 'surrounding word frequency / n-gram
        probability' the assignment asks for. Note the back-off is always
        keyed on `word` (the term whose plausibility we are estimating),
        never on the fixed context word -- otherwise a rare candidate with
        no bigram history would incorrectly inherit a common neighbour's
        high frequency."""
        prev_counts = self.bigrams.get(prev_word)
        if not prev_counts:
            return self.unigram_prob(word)
        total = sum(prev_counts.values())
        return (prev_counts.get(word, 0) + 1) / (total + len(self.vocab))

    def right_context_prob(self, candidate: str, next_word: str) -> float:
        """P(next_word | candidate) using candidate's own outgoing bigram
        distribution when available, falling back to the candidate's
        unigram probability (NOT the next word's) when candidate never
        appears as a bigram key -- this keeps rare candidates penalized
        appropriately instead of inheriting a common neighbour's frequency."""
        cand_counts = self.bigrams.get(candidate)
        if not cand_counts:
            return self.unigram_prob(candidate)
        total = sum(cand_counts.values())
        return (cand_counts.get(next_word, 0) + 1) / (total + len(self.vocab))


# Common real-word confusion sets: words that are *individually* valid
# dictionary words but are frequently swapped for each other. Even though
# each spelling passes a plain dictionary check, context should still be
# used to flag the wrong member of the set.
CONFUSION_SETS = [
    {"there", "their", "theyre"},
    {"your", "youre"},
    {"its", "its"},  # kept simple; real "it's" needs apostrophe handling
    {"then", "than"},
    {"to", "too", "two"},
    {"affect", "effect"},
    {"weather", "whether"},
    {"were", "where", "wear"},
    {"accept", "except"},
    {"loose", "lose"},
]
CONFUSION_LOOKUP: dict[str, set[str]] = {}
for group in CONFUSION_SETS:
    for w in group:
        CONFUSION_LOOKUP[w] = group


# ---------------------------------------------------------------------------
# 3. Candidate generation & ranking
# ---------------------------------------------------------------------------
class SpellChecker:
    def __init__(self, lm: LanguageModel | None = None, max_edit_distance: int = 2):
        self.lm = lm or LanguageModel()
        self.max_edit_distance = max_edit_distance

    def candidates(self, word: str, limit: int = 5):
        """Scan the dictionary and return the `limit` best (word, edit_dist)
        pairs within `self.max_edit_distance`, cheapest distance first and
        ties broken by corpus frequency (a very common word is a more
        plausible typo target than a rare one at the same distance)."""
        w = word.lower()
        found = []
        for cand in self.lm.vocab:
            # quick length-based prune before running full DP
            if abs(len(cand) - len(w)) > self.max_edit_distance:
                continue
            d = bounded_levenshtein(w, cand, self.max_edit_distance)
            if d is not None:
                found.append((cand, d))
        found.sort(key=lambda cw: (cw[1], -self.lm.freq.get(cw[0], 0)))
        return found[:limit]

    def best_correction(self, word: str, prev_word: str | None, next_word: str | None):
        """Return (correction, reason_dict) for a single (possibly
        misspelled) word, given its left and right neighbours for context.
        """
        w = word.lower()

        # Case A: word is already in the dictionary.
        if w in self.lm.vocab:
            # Still check the confusion-set case: is there a same-sound /
            # same-edit-distance-1 alternative that fits the context far
            # better? (context-aware improvement requested by the brief)
            group = CONFUSION_LOOKUP.get(w)
            if group and prev_word:
                prev_counts = self.lm.bigrams.get(prev_word.lower())
                # Only trust the bigram signal for a confusion-set swap when
                # there's enough data behind it (avoids noisy sparse counts
                # flipping a correct word in an already-ungrammatical
                # fragment, e.g. "might of to reschedule").
                if prev_counts and sum(prev_counts.values()) >= 10:
                    scores = {alt: self.lm.bigram_prob(prev_word.lower(), alt) for alt in group}
                    best_alt = max(scores, key=scores.get)
                    if best_alt != w and scores[best_alt] > scores[w] * 6:
                        return best_alt, {
                            "type": "context_real_word_error",
                            "prev_word": prev_word,
                            "bigram_prob_chosen": scores[best_alt],
                            "bigram_prob_original": scores[w],
                        }
            return w, {"type": "unchanged"}

        # Case B: word is out-of-dictionary -> generate edit-distance candidates.
        cands = self.candidates(w, limit=5)
        if not cands:
            return w, {"type": "no_candidate_found"}

        min_dist = cands[0][1]
        top = [c for c in cands if c[1] == min_dist]

        if len(top) == 1:
            return top[0][0], {
                "type": "edit_distance",
                "edit_distance": min_dist,
                "candidates_considered": [c[0] for c in cands],
            }

        # Tie among several equally-close candidates -> break the tie with
        # context (n-gram / bigram probability), i.e. the context-aware
        # improvement requested by the assignment.
        scored = []
        for cand, dist in top:
            # Context score: average bigram probability from the left and
            # right neighbour (the "surrounding word frequency / n-gram
            # probability" signal the brief asks for).
            ctx_terms = []
            if prev_word:
                ctx_terms.append(self.lm.bigram_prob(prev_word.lower(), cand))
            if next_word:
                ctx_terms.append(self.lm.right_context_prob(cand, next_word.lower()))
            ctx_score = sum(ctx_terms) / len(ctx_terms) if ctx_terms else 0.0

            # Blend with the candidate's raw unigram frequency so that, when
            # the local bigram context is too sparse to be decisive (both
            # candidates are context-neutral), the overall more common word
            # in English still wins the tie -- e.g. "receive" over "relieve"
            # after "I recieve ...".
            uni_score = self.lm.unigram_prob(cand)
            score = 0.6 * ctx_score + 0.4 * uni_score
            scored.append((cand, score))
        scored.sort(key=lambda cs: -cs[1])
        best_word, best_score = scored[0]
        return best_word, {
            "type": "context_aware_tiebreak",
            "edit_distance": min_dist,
            "tied_candidates": [c for c, _ in top],
            "ngram_scores": {c: s for c, s in scored},
        }


# ---------------------------------------------------------------------------
# 4. Tokenisation + full-text correction with a diff report
# ---------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[A-Za-z']+|[^A-Za-z'\s]|\s+")


def tokenize_keep_all(text: str):
    """Tokenize while keeping whitespace/punctuation tokens so the original
    text can be reconstructed exactly, just with corrected words swapped in.
    """
    return TOKEN_RE.findall(text)


def restore_case(original: str, corrected: str) -> str:
    if original.isupper() and len(original) > 1:
        return corrected.upper()
    if original[:1].isupper():
        return corrected[:1].upper() + corrected[1:]
    return corrected


def correct_text(text: str, checker: SpellChecker | None = None):
    """Correct `text` end to end.

    Returns
    -------
    corrected_text : str
    diff : list[dict]  -- one entry per changed word with reason metadata
    """
    checker = checker or SpellChecker()
    tokens = tokenize_keep_all(text)
    word_idx = [i for i, t in enumerate(tokens) if re.fullmatch(r"[A-Za-z']+", t)]

    diff = []
    out_tokens = list(tokens)

    def neighbor_word(pos_in_wordlist, offset):
        j = pos_in_wordlist + offset
        if 0 <= j < len(word_idx):
            tok = tokens[word_idx[j]]
            return tok
        return None

    for k, ti in enumerate(word_idx):
        original = tokens[ti]
        clean = original.strip("'")
        if not clean.isalpha():
            continue
        prev_w = neighbor_word(k, -1)
        next_w = neighbor_word(k, 1)
        correction, reason = checker.best_correction(clean, prev_w, next_w)
        if correction.lower() != clean.lower():
            cased = restore_case(original, correction)
            out_tokens[ti] = cased
            diff.append({
                "original": original,
                "corrected": cased,
                "reason": reason,
            })

    corrected_text = "".join(out_tokens)
    return corrected_text, diff


if __name__ == "__main__":
    sample = "I recieve you're email and there going to the meating tommorow."
    text, changes = correct_text(sample)
    print("ORIGINAL :", sample)
    print("CORRECTED:", text)
    for c in changes:
        print(" -", c["original"], "->", c["corrected"], "|", c["reason"]["type"])

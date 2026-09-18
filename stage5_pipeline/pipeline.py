"""
Stage 5 -- Full Pipeline Assembly
==================================

Wires Stages 1 -> 4 into a single function:

    process(raw_text) -> {
        corrected_text, spelling_diff,
        parse_trees,                # per-sentence spaCy dep tree + CFG tree
        semantic_frames,            # per-sentence agent/action/patient/entities/wsd
        discourse_relations,
        coref_chains,
        pragmatics,
    }

Also provides `run_on_folder`, which processes every .txt file in a folder
of "messy" real-world text samples and produces the one summary table the
assignment asks for (per sample: # spelling corrections, # entities found,
# coref chains resolved, # discourse relations tagged).
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage1_spellcheck.spellcheck import correct_text, SpellChecker
from stage2_syntax.syntax import process_syntax
from stage3_semantics.semantics import process_semantics
from stage4_discourse.discourse import process_discourse

_CHECKER = None


def _get_checker():
    global _CHECKER
    if _CHECKER is None:
        _CHECKER = SpellChecker()
    return _CHECKER


def process(raw_text: str) -> dict:
    """The single end-to-end function requested by the brief:
    process(raw_text) -> corrected_text, parse_trees, semantic_frames,
    discourse_relations, coref_chains."""

    # Stage 1 -- spell checking
    corrected_text, spelling_diff = correct_text(raw_text, _get_checker())

    # Stage 2 -- syntactic processing (runs on the corrected text)
    parse_trees = process_syntax(corrected_text)

    # Stage 3 -- semantic analysis (runs on the corrected text)
    semantic_frames = process_semantics(corrected_text)

    # Stage 4 -- discourse & pragmatics (runs on the corrected text so
    # coreference / connective detection isn't thrown off by typos)
    discourse = process_discourse(corrected_text)

    return {
        "raw_text": raw_text,
        "corrected_text": corrected_text,
        "spelling_diff": spelling_diff,
        "parse_trees": parse_trees,
        "semantic_frames": semantic_frames,
        "discourse_relations": discourse["discourse_relations"],
        "coref_chains": discourse["coref_chains"],
        "pragmatics": discourse["pragmatics"],
    }


def summarize(sample_name: str, result: dict) -> dict:
    total_entities = sum(len(f["entities"]) for f in result["semantic_frames"])
    return {
        "sample": sample_name,
        "# spelling_corrections": len(result["spelling_diff"]),
        "# entities_found": total_entities,
        "# coref_chains_resolved": sum(
            1 for c in result["coref_chains"] if c["antecedent"] is not None
        ),
        "# discourse_relations_tagged": len(result["discourse_relations"]),
        "# sentences": len(result["parse_trees"]),
        "# cfg_parses_succeeded": sum(1 for p in result["parse_trees"] if p["cfg"]["parsed"]),
    }


def run_on_folder(folder: str, output_dir: str = "outputs"):
    """Process every .txt file in `folder`, save full per-sample JSON
    output, and return the summary table (list of dicts) requested by the
    assignment for the 10 test samples."""
    folder_path = Path(folder)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    files = sorted(folder_path.glob("*.txt"))
    for f in files:
        raw = f.read_text(encoding="utf-8").strip()
        if not raw:
            continue
        result = process(raw)

        # save full structured output per sample
        with open(out_path / f"{f.stem}.json", "w") as jf:
            json.dump(result, jf, indent=2, default=str)

        summary_rows.append(summarize(f.name, result))

    return summary_rows


def print_summary_table(rows: list[dict]):
    if not rows:
        print("No samples processed.")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data" / "test_samples"
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    rows = run_on_folder(str(data_dir), str(output_dir))
    print_summary_table(rows)

    # also dump the summary table itself
    with open(output_dir / "summary_table.json", "w") as f:
        json.dump(rows, f, indent=2)

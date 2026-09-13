"""Score aligned translation files with corpus BLEU and classic chrF."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import sacrebleu
from sacrebleu.metrics import BLEU, CHRF


def read_lines(path: Path) -> tuple[list[str], str]:
    raw = path.read_bytes()
    # Keep empty hypotheses in place: removing them would shift alignment.
    lines = raw.decode("utf-8-sig").splitlines()
    return lines, hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypotheses", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--chrf-word-order", type=int, choices=(0, 2), default=0,
        help="0: classic chrF (default); 2: chrF++",
    )
    args = parser.parse_args()
    if args.output and args.output.resolve() in {
        args.hypotheses.resolve(), args.references.resolve()
    }:
        parser.error("output must not overwrite either input file")
    hypotheses, hypothesis_hash = read_lines(args.hypotheses)
    references, reference_hash = read_lines(args.references)
    if not references:
        parser.error("cannot score an empty corpus")
    if len(hypotheses) != len(references):
        parser.error("hypothesis and reference files must have equal line counts")
    if any(not reference.strip() for reference in references):
        parser.error("reference lines must not be blank")

    bleu = BLEU(
        lowercase=False, tokenize="13a", smooth_method="exp",
        effective_order=False,
    )
    chrf = CHRF(
        char_order=6, word_order=args.chrf_word_order, beta=2,
        lowercase=False, whitespace=False, eps_smoothing=False,
    )
    reference_streams = [references]
    bleu_score = bleu.corpus_score(hypotheses, reference_streams).score
    chrf_score = chrf.corpus_score(hypotheses, reference_streams).score
    result = {
        "examples": len(references),
        "bleu": bleu_score,
        "chrf" if args.chrf_word_order == 0 else "chrf++": chrf_score,
        "bleu_signature": str(bleu.get_signature()),
        "chrf_signature": str(chrf.get_signature()),
        "score_scale": "0-100",
        "sacrebleu_version": sacrebleu.__version__,
        "hypotheses_sha256": hypothesis_hash,
        "references_sha256": reference_hash,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .data import ParallelContextDataset
from .llm_judge import (
    JUDGE_PROMPT_VERSION,
    judge_prompt,
    judge_prompt_sha256,
    make_record,
    parse_judgment,
    random_sample_indices,
    validate_alignment,
)
from .llm_runner import LLMProviderConfig, OpenAICompatibleClient, extract_completion
from .significance import read_prediction_jsonl


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run order-swapped LLM MT judgments")
    parser.add_argument("--provider-config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--sample-seed", type=int, default=20260912)
    args = parser.parse_args()

    config = LLMProviderConfig.from_json(args.provider_config)
    if config.temperature != 0.0:
        raise ValueError("LLM judgment requires temperature 0 for stability")
    baseline_path, system_path = Path(args.baseline), Path(args.system)
    baseline = read_prediction_jsonl(baseline_path)
    system = read_prediction_jsonl(system_path)
    dataset = ParallelContextDataset(args.data_dir, args.split)
    examples = dataset.examples
    validate_alignment(examples, baseline, system)
    if args.max_examples is not None and args.sample_size is not None:
        raise ValueError("use either max-examples or sample-size, not both")
    if args.max_examples is not None:
        if args.max_examples <= 0:
            raise ValueError("max-examples must be positive")
        examples = examples[: args.max_examples]
        baseline = type(baseline)(
            baseline.hypotheses[: args.max_examples],
            baseline.references[: args.max_examples],
            baseline.line_numbers[: args.max_examples],
        )
        system = type(system)(
            system.hypotheses[: args.max_examples],
            system.references[: args.max_examples],
            system.line_numbers[: args.max_examples],
        )
    elif args.sample_size is not None:
        indices = random_sample_indices(len(examples), args.sample_size, args.sample_seed)
        examples = [examples[index] for index in indices]
        baseline = type(baseline)(
            tuple(baseline.hypotheses[index] for index in indices),
            tuple(baseline.references[index] for index in indices),
            tuple(baseline.line_numbers[index] for index in indices),
        )
        system = type(system)(
            tuple(system.hypotheses[index] for index in indices),
            tuple(system.references[index] for index in indices),
            tuple(system.line_numbers[index] for index in indices),
        )
    validate_alignment(examples, baseline, system)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[tuple[int, str]] = set()
    if output.is_file():
        completed = {
            (int(row["line_number"]), str(row["orientation"]))
            for row in (
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }

    manifest = {
        "base_url": config.base_url,
        "model_requested": config.model,
        "temperature": config.temperature,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "prompt_sha256": judge_prompt_sha256(),
        "baseline_sha256": _sha256(baseline_path),
        "system_sha256": _sha256(system_path),
        "split": args.split,
        "examples": len(examples),
        "sampling": (
            {"kind": "random", "seed": args.sample_seed, "size": args.sample_size}
            if args.sample_size is not None
            else {"kind": "prefix", "size": args.max_examples}
            if args.max_examples is not None
            else {"kind": "full"}
        ),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = output.parent / f"{output.stem}_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "base_url", "model_requested", "temperature", "prompt_version",
            "prompt_sha256", "baseline_sha256", "system_sha256", "split", "examples", "sampling",
        ):
            if previous.get(key) != manifest[key]:
                raise RuntimeError(f"resume mismatch for {key}")
    elif output.is_file():
        raise RuntimeError("refusing to resume output without its run manifest")
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    client = OpenAICompatibleClient(config)
    with output.open("a", encoding="utf-8") as handle:
        for index, example in enumerate(examples):
            hypotheses = {
                "baseline": baseline.hypotheses[index],
                "system": system.hypotheses[index],
            }
            for orientation, labels in (
                ("baseline_first", ("baseline", "system")),
                ("system_first", ("system", "baseline")),
            ):
                if (example.line_number, orientation) in completed:
                    continue
                prompt = judge_prompt(example, hypotheses[labels[0]], hypotheses[labels[1]])
                last_error: Exception | None = None
                for _ in range(3):
                    response = client.complete(prompt)
                    try:
                        judgment = parse_judgment(extract_completion(response))
                        break
                    except ValueError as error:
                        last_error = error
                else:
                    raise RuntimeError("judge returned invalid JSON three times") from last_error
                record = make_record(
                    example=example,
                    orientation=orientation,
                    judgment=judgment,
                    response=response,
                    requested_model=config.model,
                )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .data import ParallelContextDataset
from .llm_runner import (
    LLMProviderConfig,
    OpenAICompatibleClient,
    extract_completion,
    prompt_sha256,
    render_prompt,
    response_record,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate archived zero-shot LLM translations"
    )
    parser.add_argument("--provider-config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = LLMProviderConfig.from_json(args.provider_config)
    template = Path(args.prompt).read_text(encoding="utf-8")
    template_hash = prompt_sha256(template)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[int] = set()
    if output.is_file():
        completed = {
            int(json.loads(line)["line_number"])
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    manifest = {
        "base_url": config.base_url,
        "model_requested": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "prompt_sha256": template_hash,
        "split": args.split,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = output.parent / f"{output.stem}_manifest.json"
    if output.is_file() and not manifest_path.is_file():
        raise RuntimeError("refusing to resume output without its run manifest")
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("base_url", "model_requested", "temperature", "max_tokens", "prompt_sha256", "split"):
            if previous.get(key) != manifest[key]:
                raise RuntimeError(f"resume mismatch for {key}")
        manifest = previous
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    client = OpenAICompatibleClient(config)
    dataset = ParallelContextDataset(args.data_dir, args.split)
    with output.open("a", encoding="utf-8") as handle:
        for example in dataset.examples:
            if example.line_number in completed:
                continue
            response = client.complete(render_prompt(template, example))
            record = response_record(
                example,
                extract_completion(response),
                response,
                config.model,
                template_hash,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()

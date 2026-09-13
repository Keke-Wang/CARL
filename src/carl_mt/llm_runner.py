from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import TranslationExample


@dataclass(frozen=True)
class LLMProviderConfig:
    base_url: str
    model: str
    api_key_env: str
    temperature: float = 0.0
    max_tokens: int = 128
    timeout_seconds: int = 90
    retries: int = 4

    @classmethod
    def from_json(cls, path: str | Path) -> "LLMProviderConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        config = cls(**raw)
        if not config.base_url.startswith("https://"):
            raise ValueError("base_url must use HTTPS")
        if "REPLACE_" in config.base_url or "REPLACE_" in config.model:
            raise ValueError("replace placeholder base_url and model before running")
        if config.retries < 1:
            raise ValueError("retries must be at least one")
        return config


def render_prompt(template: str, example: TranslationExample) -> str:
    try:
        return template.format(title=example.title, context=example.context)
    except KeyError as error:
        raise ValueError(f"unsupported prompt placeholder: {error}") from error


def prompt_sha256(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def extract_completion(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("response lacks choices[0].message.content") from error
    if not isinstance(content, str) or not content.strip():
        raise ValueError("provider returned an empty completion")
    return content.strip()


class OpenAICompatibleClient:
    """Minimal client for a provider's documented chat-completions endpoint."""

    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config
        self.api_key = os.environ.get(config.api_key_env)
        if not self.api_key:
            raise RuntimeError(f"environment variable {config.api_key_env} is unset")

    def complete(self, prompt: str) -> dict[str, Any]:
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = json.dumps(
            {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        last_error: Exception | None = None
        for attempt in range(self.config.retries):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout_seconds
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
                last_error = error
                if attempt + 1 < self.config.retries:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError("provider request failed after retries") from last_error


def response_record(
    example: TranslationExample,
    completion: str,
    response: dict[str, Any],
    model: str,
    prompt_hash: str,
) -> dict[str, Any]:
    return {
        "line_number": example.line_number,
        "model": response.get("model", model),
        "response_id": response.get("id"),
        "created": response.get("created"),
        "system_fingerprint": response.get("system_fingerprint"),
        "prompt_sha256": prompt_hash,
        "hypothesis": completion,
        "reference": example.target,
        "usage": response.get("usage"),
    }


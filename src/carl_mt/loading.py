from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_model_and_tokenizer(
    model_name_or_path: str,
    context_token: str,
) -> tuple[Any, Any, int]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if context_token == tokenizer.eos_token:
        added = 0
    else:
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": [context_token]}
        )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path,
        attn_implementation="eager",
    )
    if added:
        model.resize_token_embeddings(len(tokenizer))
    token_id = tokenizer.convert_tokens_to_ids(context_token)
    if token_id == tokenizer.unk_token_id:
        raise ValueError(f"context token {context_token!r} is unknown")
    return model, tokenizer, int(token_id)

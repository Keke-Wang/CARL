from __future__ import annotations

import json
import os
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_reproducible_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def runtime_manifest() -> dict[str, Any]:
    try:
        import sacrebleu
        import transformers

        sacrebleu_version = sacrebleu.__version__
        transformers_version = transformers.__version__
    except ImportError:
        sacrebleu_version = "unavailable"
        transformers_version = "unavailable"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers_version,
        "sacrebleu": sacrebleu_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
    }


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


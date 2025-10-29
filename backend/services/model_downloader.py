from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

HF_CACHE = Path(os.environ.get("HF_HOME", Path.home()/".cache"/"huggingface"))
MODEL_CACHE = Path(__file__).resolve().parents[1] / "model_cache"
MODEL_CACHE.mkdir(parents=True, exist_ok=True)

try:
    from huggingface_hub import hf_hub_download  # type: ignore
except Exception:
    hf_hub_download = None  # type: ignore


def ensure_hf_file(repo_id: str, filename: str, revision: Optional[str] = None) -> Optional[Path]:
    """Download a single file from Hugging Face Hub into local cache and return its path.
    Returns None if huggingface_hub is not installed.
    """
    if hf_hub_download is None:
        return None
    local = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, local_dir=str(MODEL_CACHE))
    return Path(local)


def get_cache_dir() -> Path:
    return MODEL_CACHE

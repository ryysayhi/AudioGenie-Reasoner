from __future__ import annotations
import os
import json
from typing import Any, Optional


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resolve_audio_path(audio_id_value: str, audio_dir: Optional[str] = None) -> str:
    if not audio_id_value:
        raise ValueError("audio_id or audio_path is missing")
    if audio_dir:
        return os.path.join(audio_dir, os.path.basename(audio_id_value))
    return audio_id_value
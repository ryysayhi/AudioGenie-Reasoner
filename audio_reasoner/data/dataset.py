from __future__ import annotations
from typing import Any, Dict, Iterable, List

from ..utils.io import load_json


class MMARDataset:
    """最简单的 JSON 列表数据读取器。"""

    def __init__(self, json_path: str) -> None:
        items = load_json(json_path)
        assert isinstance(items, list), "输入 JSON 顶层应为列表。"
        self.items: List[Dict[str, Any]] = items

    def __len__(self):
        return len(self.items)

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for ex in self.items:
            yield ex
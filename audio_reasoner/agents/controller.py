from __future__ import annotations
import json
from typing import Any, Dict, List

from .prompts import (
    PLANNING_SYS,
    PLANNING_USER_TMPL,
    INTERACTION_SYS,
    INTERACTION_USER_TMPL,
    ANSWERING_SYS,
    ANSWERING_USER_TMPL,
)
from .structures import CaptionDoc
from ..models.deepseek_client import DeepSeekClient
from ..models.dasheng_allm import DashengALLM


class Agents:
    def __init__(self, deepseek: DeepSeekClient, allm: DashengALLM) -> None:
        self.deepseek = deepseek
        self.allm = allm

    def plan(self, question: str, choices: List[str], caption: CaptionDoc, history: List[Dict[str, Any]]) -> dict:
        if not history:
            # 如果 history 为空，插入默认值
            history = [{"sufficient": False, "reason": "Initial check", "confidence_estimate": 0.0, "missing_details": [], "proposed_aug_types": [], "proposed_questions": []}]
    
        user = PLANNING_USER_TMPL.format(
            question=question,
            choices=json.dumps(choices, ensure_ascii=False, indent=2),
            caption_doc=json.dumps(caption.to_dict(), ensure_ascii=False, indent=2),
            history=history
        )
        return self.deepseek.chat_for_json(PLANNING_SYS, user)

    def interact(self, caption: CaptionDoc, history: List[Dict[str, Any]]) -> dict:
        user = INTERACTION_USER_TMPL.format(
            caption_doc=json.dumps(caption.to_dict(), ensure_ascii=False, indent=2),
            history=json.dumps(history, ensure_ascii=False, indent=2),
        )
        return self.deepseek.chat_for_json(INTERACTION_SYS, user)

    def answer(self, question: str, choices: List[str], caption: CaptionDoc) -> dict:
        user = ANSWERING_USER_TMPL.format(
            question=question,
            choices=json.dumps(choices, ensure_ascii=False, indent=2),
            caption_doc=json.dumps(caption.to_dict(), ensure_ascii=False, indent=2),
        )
        return self.deepseek.chat_for_json(ANSWERING_SYS, user)
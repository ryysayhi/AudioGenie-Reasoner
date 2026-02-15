from __future__ import annotations
from typing import Any, Dict, List, Tuple

from ..agents.controller import Agents
from ..agents.structures import CaptionDoc
from ..augmentation.runner import AugmentationRunner
from ..utils.normalization import normalize_choice_prediction
from ..utils.prompting import coerce_json

import time as _t

class MultiAgentAudioReasoner:
    def __init__(self, agents: Agents, max_iters: int = 3, sleep_between_calls: float = 0.0, whisper=None) -> None:
        self.agents = agents
        self.max_iters = max_iters
        self.sleep = sleep_between_calls
        self.aug = AugmentationRunner(agents.allm, sleep_between_calls, whisper=whisper)

    def initial_caption(self, audio_path: str) -> CaptionDoc:
        cap = self.agents.allm.caption(audio_path)
        return CaptionDoc(summary=cap.strip())

    def iterate(self, question: str, choices: List[str], audio_path: str) -> Tuple[CaptionDoc, List[Dict[str, Any]], bool]:
        caption = self.initial_caption(audio_path)
        history: List[Dict[str, Any]] = []

        for it in range(1, self.max_iters + 1):
            t0 = _t.time()
            plan_raw = self.agents.plan(question, choices, caption, history)
            print(f"[plan i{it}] {(_t.time()-t0):.2f}s")

            if "raw" in plan_raw:
                import json
                plan_raw = json.loads(plan_raw["raw"])
            if isinstance(plan_raw, str):
                plan_raw = json.loads(plan_raw)

            plan_js = coerce_json(plan_raw, {
                "sufficient": False,
                "reason": "",
                "confidence_estimate": 0.0,
                "missing_details": [],
                "proposed_aug_types": [],
                "proposed_questions": [],
            })
            plan_js.setdefault("sufficient", False)
            plan_js.setdefault("reason", "")
            plan_js.setdefault("confidence_estimate", 0.0)
            plan_js.setdefault("missing_details", [])
            plan_js.setdefault("proposed_aug_types", [])
            plan_js.setdefault("proposed_questions", [])

            sufficient = bool(plan_js.get("sufficient"))
            reason = plan_js.get("reason", "")
            conf = float(plan_js.get("confidence_estimate", 0.0))
            missing = plan_js.get("missing_details", []) or []
            proposed = plan_js.get("proposed_aug_types", []) or []
            proposed_questions = plan_js.get("proposed_questions", []) or []

            history.append({
                "iteration": it,
                "sufficient": sufficient,
                "reason": reason,
                "confidence_estimate": conf,
                "missing_details": missing,
                "proposed_aug_types": proposed,
                "proposed_questions": proposed_questions,
            })
            if sufficient:
                return caption, history, True

            # === augmentation ===
            t1 = _t.time()
            inter_js = self.agents.interact(caption, history)
            print(f"[interact i{it}] {(_t.time()-t1):.2f}s")

            inter_raw = inter_js
            inter_js = coerce_json(inter_raw, {"plan": []})
            if "plan" not in inter_js and "augmentation_plan" in inter_js:
                inter_js["plan"] = [inter_js["augmentation_plan"]]
            if isinstance(inter_js.get("plan"), dict):
                inter_js["plan"] = [inter_js["plan"]]
            inter_js.setdefault("plan", [])

            t2 = _t.time()
            aug_doc = self.aug.run(audio_path, inter_js)
            print(f"[augment i{it}] {(_t.time()-t2):.2f}s")

            caption = CaptionDoc.merge(caption, aug_doc)

        return caption, history, False


    def answer(self, question: str, choices: List[str], caption: CaptionDoc) -> Dict[str, Any]:
        ans_raw = self.agents.answer(question, choices, caption)
        ans_js = coerce_json(ans_raw, {"answer": "", "confidence": 0.0, "reasoning": ""})
        raw_answer = (ans_js.get("answer") or "").strip()

        final_choice, tag = normalize_choice_prediction(raw_answer, choices)
        conf = float(ans_js.get("confidence", 0.0))
        reasoning = ans_js.get("reasoning", ans_js.get("raw", "")).strip()
        return {
            "answer": final_choice,
            "confidence": conf,
            "reasoning": reasoning,
            "raw_answer": raw_answer,
            "normalization": tag,
        }
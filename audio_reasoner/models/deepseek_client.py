from __future__ import annotations
import os
import json
from typing import Any, Dict, List, Optional

from ..utils.prompting import extract_json_block
from ..utils.prompting import coerce_json

class DeepSeekClient:
    def __init__(self, api_base: str = "https://aigc-api.hkust-gz.edu.cn/v1", api_key_env: str = "your_api_key", default_model: str = "gpt-4", timeout: int = 120) -> None:
        self.api_base = api_base
        self.api_key_env = api_key_env
        self.default_model = default_model
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        # key = os.environ.get(self.api_key_env, "")
        key = self.api_key_env
        if not key:
            raise RuntimeError(f"Missing API key env var: {self.api_key_env}")
        return {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}

    def chat(self, messages: List[Dict[str, Any]], model: Optional[str] = None, **extra) -> str:
        import requests
        url = f"{self.api_base}/chat/completions"
        payload = {"model": model or self.default_model, "messages": messages}
        payload.update(extra)
        r = requests.post(url, 
                          headers=self._headers(),
                          data=json.dumps(payload), 
                          timeout=self.timeout
                          )
        r.raise_for_status()
        js = r.json()
        try:
            return js["choices"][0]["message"]["content"]
        except Exception:
            return json.dumps(js)

    def chat_for_json(self, sys_prompt: str, user_prompt: str, model: Optional[str] = None, **extra) -> dict:
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
        txt = self.chat(messages, model=model, **extra)
        # js = extract_json_block(txt)
        fallback = {
            "sufficient": False,
            "reason": "",
            "confidence_estimate": 0.0,
            "missing_details": [],
            "proposed_aug_types": [],
            "proposed_questions": [],
        }
        js = coerce_json(txt, fallback)
        if js is None:
            js = {"raw": txt}
        return js

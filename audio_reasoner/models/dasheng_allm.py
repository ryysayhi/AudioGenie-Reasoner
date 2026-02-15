from __future__ import annotations
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor


class DashengALLM:
    def __init__(self, model_id: str = "mispeech/midashenglm-7b", device_map: str = "cuda", torch_dtype: Optional[torch.dtype] = None) -> None:
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map=device_map, torch_dtype=torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def _gen(self, messages: List[Dict[str, Any]], **gen_kwargs) -> str:
        model_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            add_special_tokens=True,
            return_dict=True,
        )
        model_inputs = {k: (v.to(self.device) if hasattr(v, 'to') else v) for k, v in model_inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**model_inputs, **gen_kwargs)
        text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return text[0]

    def caption(self, audio_path: str, extra_instruction: str = "") -> str:
        user_prompt = (
            "Listen carefully and produce a detailed audio caption." + extra_instruction
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are an expert audio captioning assistant."}]},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt}, 
                {"type": "audio", "path": audio_path}
                ]
            },
        ]
        return self._gen(messages, max_new_tokens=180)

    def audio_qa(self, audio_path: str, questions: List[str]) -> List[Dict[str, str]]:
        qa_pairs = []
        for q in questions:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You answer audio-focused questions tersely and accurately."}]},
                {"role": "user", "content": [   {"type": "text", "text": f"Question: {q}\nAnswer in one short sentence. "},
                                                {"type": "audio", "path": audio_path}
                                            ]},
            ]
            ans = self._gen(messages, max_new_tokens=180)
            qa_pairs.append({"question": q, "answer": ans.strip()})
        return qa_pairs

    def transcribe(self, audio_path: str, hint: str = "") -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Transcribe spoken words as best as you can."}]},
            {"role": "user", "content": [{"type": "text", "text": f"Provide a rough transcript with speaker turns and timestamps if possible. {hint}"}, {"type": "audio", "path": audio_path}]},
        ]
        return self._gen(messages, max_new_tokens=120)
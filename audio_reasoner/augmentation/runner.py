from __future__ import annotations
from typing import Dict
import time

from ..models.dasheng_allm import DashengALLM
from ..agents.structures import CaptionDoc


def _fmt_ts(sec: float) -> str:
    sec = max(0, int(round(sec)))
    m, s = divmod(sec, 60)
    return f"{m:02d}:{s:02d}"


def _format_whisper_segments(res: dict, max_chars: int = 1200, gap_newline: float = 0.8) -> str:
    """
    将 whisper 的 segments 渲染为带时间戳的多行文本；过长时截断。
    gap_newline: 相邻片段起始与上一片段结束的间隔超过此秒数，就加空行以便阅读。
    """
    segs = res.get("segments") or []
    if not segs:  # 没有 segments，回退到纯 text
        return (res.get("text") or "").strip() or "[inaudible]"

    lines, total = [], 0
    prev_end = None
    for seg in segs:
        # 兼容 dict 或对象
        start = getattr(seg, "start", None) if not isinstance(seg, dict) else seg.get("start", 0.0)
        end   = getattr(seg, "end", None)   if not isinstance(seg, dict) else seg.get("end", 0.0)
        text  = getattr(seg, "text", None)  if not isinstance(seg, dict) else seg.get("text", "")
        if text is None:
            continue
        text = text.strip()
        if not text:
            continue
        if prev_end is not None and start is not None and (start - prev_end) >= gap_newline:
            lines.append("")  # 插入空行表示明显停顿
        ts = f"[{_fmt_ts(start)}-{_fmt_ts(end)}]" if (start is not None and end is not None) else ""
        line = f"{ts} {text}".strip()
        lines.append(line)
        total += len(line) + 1
        prev_end = end if end is not None else prev_end
        if total >= max_chars:
            lines.append("...")  # 太长就截断
            break
    return "\n".join(lines) if lines else "[inaudible]"


class AugmentationRunner:
    def __init__(self, allm: DashengALLM, sleep_between_calls: float = 0.0, whisper=None) -> None:
        self.allm = allm
        self.sleep = sleep_between_calls
        self.whisper = whisper

    def run(self, audio_path: str, plan: Dict) -> CaptionDoc:
        aug_doc = CaptionDoc(summary="")
        for step in plan.get("plan", []):
            typ = (step.get("type") or "").lower()
            instr = step.get("instructions") or ""
            questions = step.get("questions") or []

            # 轻量日志，帮助你定位是否卡在这里
            print(f"[augment step] type={typ} q_count={len(questions)}")

            if typ == "audio_qa":
                qa = self.allm.audio_qa(audio_path, questions or ["What detail is missing?"])
                aug_doc.qa.extend(qa)
            elif typ == "re_caption":
                extra = f"Focus on: {instr}" if instr else ""
                rc = self.allm.caption(audio_path, extra_instruction=extra)
                aug_doc.details.append(rc.strip())
            elif typ == "transcription":
                try:
                    if self.whisper is not None:
                        res = self.whisper.transcribe(audio_path)
                        # tr = (res.get("text") or "").strip() or "[inaudible]"
                        tr = _format_whisper_segments(res, max_chars=1200, gap_newline=0.8)
                    else:
                        # 没有 whisper 实例就回退 Dasheng
                        tr = self.allm.transcribe(audio_path, hint=instr)
                except Exception as e:
                    print(f"[warn] whisper transcribe failed: {type(e).__name__}: {e}  → fallback to ALLM")
                    tr = self.allm.transcribe(audio_path, hint=instr)
                if tr:
                    aug_doc.transcript_excerpt = (aug_doc.transcript_excerpt or "") + ("\n\n" if aug_doc.transcript_excerpt else "") + tr.strip()
            elif typ == "timestamping":
                qs = questions or ["When (mm:ss) does the key event occur?"]
                qa = self.allm.audio_qa(audio_path, qs)
                aug_doc.qa.extend(qa)
            else:
                aug_doc.details.append(f"[Skipped unknown augmentation type: {typ}]")

            if self.sleep:
                time.sleep(self.sleep)
        return aug_doc
import json
import re
from typing import List, Optional, Any, Dict


def build_prompt(question: str, choices: List[str]) -> str:
    lines = [f"Question: {question}", "Answer List = ["]
    for ch in choices:
        lines.append(f'    "{ch}",')
    lines.append("]")
    return "\n".join(lines)


def extract_json_block(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    block = m.group(0)
    # print("block:", block)
    try:
        return json.loads(block)
    except Exception:
        return None


def coerce_json(text: Any, fallback: Dict | None = None) -> Dict:
    # 1) 已经是 dict 直接返回
    if isinstance(text, dict):
        return text

    s = (text or "").strip()
    if not s:
        return (fallback.copy() if isinstance(fallback, dict) else {"raw": ""})

    # 2) 去掉代码围栏
    s2 = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.DOTALL).strip()

    # 3) 直接 parse
    try:
        obj = json.loads(s2)
        return obj if isinstance(obj, dict) else {"data": obj}
    except Exception:
        pass

    # 4) 抓取第一个 JSON 对象/数组再 parse
    m = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", s2)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {"data": obj}
        except Exception:
            pass

    # 5) 全部失败：回退
    if isinstance(fallback, dict):
        fb = fallback.copy()
        fb["raw"] = s[:2000]
        return fb
    return {"raw": s[:2000]}
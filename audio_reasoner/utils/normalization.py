import math
import re
from typing import List, Tuple, Optional


def letter_to_index(letter: str) -> Optional[int]:
    letter = letter.strip().upper()
    if len(letter) == 1 and 'A' <= letter <= 'Z':
        return ord(letter) - ord('A')
    return None


def normalize_choice_prediction(raw_pred: str, choices: List[str]) -> Tuple[str, str]:
    if not choices:
        return raw_pred.strip(), "no_choices"

    pred = raw_pred.strip()

    # 1) exact
    for c in choices:
        if pred == c:
            return c, "exact"

    # 2) case-insensitive exact
    for c in choices:
        if pred.lower() == c.lower():
            return c, "case_exact"

    # 3) single letter A/B/C
    m = re.match(r"^([A-Za-z])\.?$", pred)
    if m:
        idx = letter_to_index(m.group(1))
        if idx is not None and 0 <= idx < len(choices):
            return choices[idx], "letter"

    # 4) numeric suffix mapping (e.g., "14" → match "A.14")
    digits = re.findall(r"\d+", pred)
    if digits:
        for d in digits:
            for c in choices:
                if re.search(rf"(^|[^\d]){d}($|[^\d])", c):
                    return c, "digits_in_choice"

    # 5) substring containment
    for c in choices:
        if pred in c:
            return c, "substring"
        if c in pred:
            return c, "superset"

    # 6) token overlap fallback
    def score(a: str, b: str) -> float:
        sa = set(re.findall(r"\w+", a.lower()))
        sb = set(re.findall(r"\w+", b.lower()))
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        denom = math.sqrt(len(sa) * len(sb))
        return inter / denom if denom else 0.0

    best = max(choices, key=lambda c: score(pred, c))
    return best, "overlap_fallback"
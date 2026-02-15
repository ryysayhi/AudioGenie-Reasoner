from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import copy


@dataclass
class CaptionDoc:
    summary: str
    details: List[str] = field(default_factory=list)
    qa: List[Dict[str, str]] = field(default_factory=list)
    transcript_excerpt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "details": self.details,
            "qa": self.qa,
            "transcript_excerpt": self.transcript_excerpt,
            "metadata": self.metadata,
        }

    @staticmethod
    def merge(base: "CaptionDoc", extra: "CaptionDoc") -> "CaptionDoc":
        merged = copy.deepcopy(base)
        if extra.summary and extra.summary not in (base.summary or ""):
            merged.details.append(f"Additional summary: {extra.summary}")
        merged.details.extend([d for d in extra.details if d not in merged.details])
        merged.qa.extend(extra.qa)
        if extra.transcript_excerpt:
            if merged.transcript_excerpt:
                merged.transcript_excerpt += "\n\n" + extra.transcript_excerpt
            else:
                merged.transcript_excerpt = extra.transcript_excerpt
        merged.metadata.update(extra.metadata or {})
        return merged
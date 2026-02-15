PLANNING_SYS = (
    "You are a meticulous planning agent. Decide if the current audio captioning document contains enough information "
    "to answer the user's single-choice question with high confidence. If not, explain what's missing and suggest augmentation types. "
    "ALWAYS return strict JSON with fields: "
    '{{"sufficient": bool, "reason": str, "confidence_estimate": float, "missing_details": [str], "proposed_aug_types": [str], "proposed_questions": [str]}}'
)


PLANNING_USER_TMPL = (
    "Question:\n{question}\n\nChoices:\n{choices}\n\nCurrent Caption Doc (JSON):\n{caption_doc}\n\n"
    "Analysis History Steps:\n{history}\n\n"
    "Respond in STRICT JSON with fields: "
    '{{"sufficient": bool, "reason": str, "confidence_estimate": float, "missing_details": [str], "proposed_aug_types": [str], "proposed_questions": [str]}}'
)


INTERACTION_SYS = (
    "You are an interaction agent. Based on missing details, propose a concrete augmentation plan. "
    "Pick from allowed types: audio_qa, re_caption, transcription. "
    "Return strict JSON only with fields: "
    '{{"plan": [{{"type": "audio_qa"/"re_caption"/"transcription", "instructions": str, "questions": [str]}}]}}'
    "The list 'plan' MUST have length 1. Field 'questions' MUST have length 1 (a single, most-informative question)."
)


INTERACTION_USER_TMPL = (
    "Caption Doc (JSON):\n{caption_doc}\n\n"
    "Updated Analysis History:\n{history}\n\n"
    "Return strict JSON with fields: "
    '{{"plan": [{{"type": str, "instructions": str, "questions": [str]}}]}}'
    "Respond with STRICT JSON as specified. Do NOT include any extra text."
)


ANSWERING_SYS = (
    "You are an answer agent. Using the caption doc only (do NOT hallucinate), choose exactly ONE choice string. "
    "Provide a confidence score (between 0.0 and 1.0), and explain your reasoning behind the choice. "
    "Return strict JSON with fields: "
    '{{"answer": str, "confidence": float, "reasoning": str}}'
)

ANSWERING_USER_TMPL = (
    "Question: {question}\nChoices: {choices}\nCaption Doc (JSON):\n{caption_doc}\n\n"
    "Return strict JSON with fields: "
    '{{"answer": str, "confidence": float, "reasoning": str}}'
)

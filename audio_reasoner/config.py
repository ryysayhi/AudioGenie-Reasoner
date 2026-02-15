from dataclasses import dataclass


@dataclass
class AppConfig:
    base_dir: str
    input_json: str
    audio_dir: str
    output_dir: str
    output_json: str
    model_id: str = "mispeech/midashenglm-7b"
    deepseek_model: str = "gpt-4o-2024-08-06"
    whisper_model: str = "turbo"
    max_iters: int = 3
    sleep: float = 0.0

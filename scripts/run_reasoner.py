import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import os
import argparse

from audio_reasoner.config import AppConfig
from audio_reasoner.utils.io import load_json, dump_json, ensure_dir
from audio_reasoner.utils.io import resolve_audio_path
from audio_reasoner.models.dasheng_allm import DashengALLM
from audio_reasoner.models.deepseek_client import DeepSeekClient
from audio_reasoner.agents.controller import Agents
from audio_reasoner.orchestration.orchestrator import MultiAgentAudioReasoner
# from audio_reasoner.models.qwen_omni_allm import QwenOmniALLM
# from audio_reasoner.models.audios_flamingo_v3_allm import AudioFlamingoV3ALLM

import whisper

def parse_args():
    p = argparse.ArgumentParser(description="Multi-agent audio reasoning")
    p.add_argument("--base_dir", default="/hpc2hdd/home/yrong854/jhaidata/LLM/benchmark/MMAU")
    p.add_argument("--input_json", default="mmau-test-mini.json")
    p.add_argument("--audio_dir", default="test-mini-audios")
    p.add_argument("--output_dir", default="/hpc2hdd/home/yrong854/jhaidata/LLM/agent/Audio-Agent/result")
    p.add_argument("--output_json", default="mmau_result.json")
    p.add_argument("--model_id", default="mispeech/midashenglm-7b") #  / mispeech/midashenglm-7b, Qwen/Qwen2.5-Omni-3B
    p.add_argument("--deepseek_model", default="gpt-4") # "DeepSeek-R1-671B", "gpt-3.5-turbo", "gpt-4" 
    p.add_argument("--whisper_model", default="large", help="faster-whisper model name or local path; empty disables Whisper")
    p.add_argument("--max_iters", type=int, default=3)
    p.add_argument("--sleep", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()

    cfg = AppConfig(
        base_dir=args.base_dir,
        input_json=args.input_json,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        output_json=args.output_json,
        model_id=args.model_id,
        deepseek_model=args.deepseek_model,
        whisper_model=args.whisper_model,
        max_iters=args.max_iters,
        sleep=args.sleep,
    )

    input_path = cfg.input_json if os.path.isabs(cfg.input_json) else os.path.join(cfg.base_dir, cfg.input_json)
    audio_dir = cfg.audio_dir if os.path.isabs(cfg.audio_dir) else os.path.join(cfg.base_dir, cfg.audio_dir)
    output_path = cfg.output_json if os.path.isabs(cfg.output_json) else os.path.join(cfg.output_dir, cfg.output_json)

    items = load_json(input_path)
    assert isinstance(items, list), "输入 JSON 顶层应为列表。"

    # os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

    allm = DashengALLM(model_id=cfg.model_id)
    # allm = QwenOmniALLM(model_id=cfg.model_id)
    # allm = AudioFlamingoV3ALLM(
    #     model_base="nvidia/audio-flamingo-3",
    #     use_think_mode=False,          # 和你给的示例一致
    #     think_lora_subdir="stage35",  # 若无该目录可设为 None
    #     conv_mode="auto",
    #     device="cuda",
    # )
    
    YOUR_API_KEY = "" # input your api key here, or set it in env var specified by `api_key_env`
    deepseek = DeepSeekClient(api_key_env=YOUR_API_KEY, default_model=cfg.deepseek_model)
    agents = Agents(deepseek=deepseek, allm=allm)

    wmodel = whisper.load_model(cfg.whisper_model)
    orchestrator = MultiAgentAudioReasoner(agents=agents, max_iters=cfg.max_iters, sleep_between_calls=cfg.sleep, whisper=wmodel)

    ensure_dir(cfg.output_dir)

    results = []
    for idx, ex in enumerate(items):
        try:
            question = ex.get("question", "")
            choices = ex.get("choices", [])
            audio_id_value = ex.get("audio_path") or ex.get("audio_id")
            audio_path = resolve_audio_path(audio_id_value, audio_dir)

            caption_doc, analysis_history, sufficient = orchestrator.iterate(question, choices, audio_path)
            final = orchestrator.answer(question, choices, caption_doc)
            final["iterations"] = len(analysis_history)
            final["sufficient_before_answering"] = sufficient

            out_ex = dict(ex)
            out_ex["caption_doc"] = caption_doc.to_dict()
            out_ex["analysis_history"] = analysis_history
            out_ex["final"] = final
            results.append(out_ex)

            print(f"[OK] {idx+1}/{len(items)} answer={final['answer']} conf={final['confidence']:.2f}")
        except Exception as e:
            print(f"[FAIL] {idx+1}/{len(items)} {type(e).__name__}: {e}")
            out_ex = dict(ex)
            out_ex["error"] = f"{type(e).__name__}: {e}"
            results.append(out_ex)

    dump_json(results, output_path)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()






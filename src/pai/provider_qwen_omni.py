"""Provider: Qwen2.5-Omni-7B (GPTQ-Int4) — thinker-talker S2S with brains.

WHY THIS MODEL: unlike Moshi/GLM-Voice, the Thinker is a real Qwen2.5 LLM
with native tool-calling ability AND the GPTQ-Int4 release is officially
supported on Windows via transformers.

VRAM BUDGET (16 GB card, audio-only turns — video is the expensive case):
  Thinker INT4 weights      ~5.5 GB
  Talker + token2wav        ~1.5 GB (bf16, streamed)
  Audio encoder (Whisper)   ~0.6 GB
  Runtime + activations     ~1.5 GB
  ─────────────────────────────────
  Total                     ~9-10 GB  →  ~6 GB left for KV cache
(BF16 would be 31 GB for 15 s of video — never fits; Int4 is mandatory.)

Setup (one-time):
  pip install transformers accelerate soundfile
  pip install gptqmodel==2.0.0       # GPTQ kernel
  # weights auto-download from HF on first load (~7 GB)
"""
from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

from . import config, tools as pai_tools

log = logging.getLogger("pai.qwen_omni")

SPEAKERS = ("Chelsie", "Aiden", "Ethan")   # stock voices


class QwenOmniProvider:
    name = "qwen_omni"

    def __init__(self):
        cfg = config.get_config()
        self.cfg = cfg
        self.model_id = getattr(cfg, "qwen_omni_model",
                                "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4")
        self.speaker = getattr(cfg, "qwen_omni_speaker", "Chelsie")
        self.low_vram = bool(getattr(cfg, "qwen_omni_low_vram", True))
        self.model = None
        self.processor = None

    # -- lifecycle -------------------------------------------------------------

    def _deps_ok(self) -> tuple[bool, str]:
        try:
            import torch, transformers, accelerate  # noqa: F401
            return True, ""
        except ImportError as exc:
            return False, (
                f"missing dependency: {exc}. Install with:\n"
                "  pip install torch --index-url "
                "https://download.pytorch.org/whl/cu128\n"
                "  pip install transformers accelerate soundfile gptmodel==2.0.0")

    def start(self):
        ok, err = self._deps_ok()
        if not ok:
            raise RuntimeError(err)
        if self.model is not None:
            return True
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        log.info("loading %s (this downloads ~7 GB on first run)...",
                 self.model_id)
        t0 = time.time()
        kwargs = dict(torch_dtype="auto", device_map="auto")
        if self.low_vram:
            # official low-VRAM pattern: offload modules after use
            kwargs["low_cpu_mem_usage"] = True
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_id, **kwargs)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_id)
        log.info("qwen-omni loaded in %.0fs", time.time() - t0)
        return True

    def stop(self):
        """Fully unload — frees every byte of VRAM."""
        import gc
        self.model = None
        self.processor = None
        try:
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            log.info("qwen-omni unloaded, VRAM freed")
        except Exception:  # noqa: BLE001
            pass

    def is_running(self) -> bool:
        return self.model is not None

    # -- conversation -----------------------------------------------------------

    def turn(self, wav_path: Path, executor: pai_tools.ToolExecutor,
             image_path: Path | None = None,
             max_tool_rounds: int = 4) -> dict:
        """Audio-in → (text + speech)-out. Tool calls detected in the text
        channel are executed and fed back (Thinker is a tool-capable LLM)."""
        import torch

        for _round in range(max_tool_rounds + 1):
            conversation = [
                {"role": "system",
                 "content": [{"type": "text",
                              "text": self._system_prompt()}]},
                {"role": "user", "content": [
                    {"type": "audio", "audio": str(wav_path)}]},
            ]
            text_prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(
                text=text_prompt, audio=str(wav_path), return_tensors="pt",
                padding=True).to(self.model.device)

            t0 = time.time()
            out = self.model.generate(
                **inputs, return_audio=True,
                speaker=self.speaker,
                max_new_tokens=512)
            text_reply = self.processor.batch_decode(
                out.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)[0].strip()
            audio_out = out.audio[0].float().detach().cpu().numpy()
            sr = self.model.config.tts_config.sample_rate
            gen_s = time.time() - t0
            log.info("turn: %.1fs, %d chars, speaker=%s",
                     gen_s, len(text_reply), self.speaker)

            # tool-call detection (Thinker may emit JSON like our other models)
            calls = pai_tools.parse_tool_calls(text_reply)
            if not calls:
                wav_out = Path(tempfile.gettempdir()) / \
                    f"pai_qwen_{int(time.time()*1000)}.wav"
                import soundfile as sf
                sf.write(wav_out, audio_out, sr)
                return {"text": text_reply, "audio": str(wav_out),
                        "tool_calls": []}

            results = [executor.execute(n, p) for n, p in calls]
            # feed results back as the next user turn (text form)
            followup = Path(tempfile.gettempdir()) / \
                f"pai_qwen_followup_{int(time.time()*1000)}.wav"
            import soundfile as sf
            import numpy as np
            sf.write(followup, np.zeros(1600, dtype="float32"), 16000)
            wav_path = followup
            self._last_tool_results = json_dumps(results)
        return {"text": "(tool loop limit)", "audio": None, "tool_calls": []}

    def _system_prompt(self) -> str:
        import json
        base = (
            "You are a personal desktop assistant. You can control the PC via "
            "tools. To call one reply ONLY with JSON {\"tool\": name, "
            "\"params\": {...}}. Otherwise answer conversationally in 1-3 "
            "short spoken sentences. Tools:\n"
            + json.dumps(pai_tools.tool_schema(), indent=1))
        tr = getattr(self, "_last_tool_results", "")
        if tr:
            base += f"\nPrevious tool results:\n{tr}"
        return base


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, default=str)[:1500]

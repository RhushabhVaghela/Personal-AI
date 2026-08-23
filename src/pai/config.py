"""Global configuration for PersonalAI-Assistant.

All knobs live here + an optional config.yaml the user can edit without
touching code. Provider selection is runtime-swappable.

Provider PROFILES let the same modular pipeline run fully local OR on
online APIs (for hardware without a GPU). API keys come from env vars —
never hardcode them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Asset paths (your existing resources)
# ---------------------------------------------------------------------------
VOICECHAT_DIR = Path(r"D:\hoidhxd-NVIDIA-NemotronLabs-VoiceChat-11B-GGUF")
VOICECHAT_EXE = VOICECHAT_DIR / "llama-voicechat.exe"
VOICECHAT_MODELS = VOICECHAT_DIR / "llamacpp"
GGUF_MAIN = VOICECHAT_MODELS / "nemotron_voicechat_11b-stt-llm-Q4_0.gguf"
GGUF_MMPROJ = VOICECHAT_MODELS / "mmproj-voicechat-perception-Q4_0.gguf"
GGUF_TTS = VOICECHAT_MODELS / "voicechat-tts-Q4_0.gguf"
GGUF_FUNCHEAD = VOICECHAT_MODELS / "nemotron_voicechat_11b-stt-llm-Q4_0-function-head.gguf"

WHISPER_SERVER_URL = os.environ.get("PAI_WHISPER_URL", "http://127.0.0.1:9000/v1/audio/transcriptions")
OMNIVOICE_BASE_URL = os.environ.get("PAI_OMNIVOICE_URL", "http://127.0.0.1:8889")
OMNIVOICE_TTS_ENDPOINT = "/v1/audio/speech"

COMPUTER_USE_DIR = Path(r"D:\Agents-and-other-repos\Computer-Use")
REALTIME_DUBBING_DIR = Path(r"D:\Agents-and-other-repos\Realtime Dubbing")

# ---------------------------------------------------------------------------
# Provider profiles: "local" | "online-openai" | "online-groq"
# (offline-first: 'local' is always the default)
# ---------------------------------------------------------------------------

PROFILES: dict[str, dict] = {
    "local": {
        "label": "Fully local (offline)",
        "asr_backend": "whisper_server",
        "asr_url": WHISPER_SERVER_URL,
        "asr_key_env": None,
        "llm_base_url": "http://127.0.0.1:8081/v1",
        "llm_model": "hy-mt2",
        "llm_key_env": None,
        "tts_backend": "omnivoice",          # omnivoice | edge | openai
        "offline": True,
    },
    "online-openai": {
        "label": "OpenAI (Whisper API + GPT + TTS)",
        "asr_backend": "openai",
        "asr_url": "https://api.openai.com/v1/audio/transcriptions",
        "asr_key_env": "OPENAI_API_KEY",
        "llm_base_url": "https://api.openai.com/v1",
        "llm_model": "gpt-4o-mini",
        "llm_key_env": "OPENAI_API_KEY",
        "tts_backend": "openai",
        "offline": False,
    },
    "online-groq": {
        "label": "Groq (whisper-large + Llama 3.x + browser TTS)",
        "asr_backend": "groq",
        "asr_url": "https://api.groq.com/openai/v1/audio/transcriptions",
        "asr_key_env": "GROQ_API_KEY",
        "llm_base_url": "https://api.groq.com/openai/v1",
        "llm_model": "llama-3.3-70b-versatile",
        "llm_key_env": "GROQ_API_KEY",
        "tts_backend": "browser",            # dashboard speaks via Web Speech
        "offline": False,
    },
}


@dataclass
class Config:
    # "voicechat" | "modular" | "hybrid"
    provider: str = "voicechat"
    profile: str = "local"                   # key into PROFILES
    # modular pipeline pieces (overridden by profile at runtime)
    asr_backend: str = "whisper_server"
    llm_base_url: str = "http://127.0.0.1:8081/v1"
    llm_model: str = "hy-mt2"
    tts_backend: str = "omnivoice"           # omnivoice | edge | openai | browser
    # hybrid: vision handled by
    vlm_base_url: str = "http://127.0.0.1:8082/v1"
    vlm_model: str = "qwen2-vl"
    # voicechat provider
    voicechat_use_funchead: bool = True
    voicechat_port: int = 8123
    # audio
    sample_rate: int = 16000
    push_to_talk_seconds: float = 6.0
    # hands-free (VAD) mode
    hands_free: bool = True
    vad_engine: str = "auto"                 # auto | webrtc | energy
    vad_aggressiveness: int = 2
    vad_silence_ms: int = 700                # end-of-speech hangover
    vad_min_utterance_ms: int = 300
    # wake word ("proactive audio")
    wake_phrases: list = field(default_factory=lambda: [
        "hey assistant", "assistant", "computer"])  # transcript gate
    wake_oww_models: list = field(default_factory=lambda: [
        "hey_jarvis", "alexa"])              # openWakeWord audio models
    # memory
    memory_turns: int = 24                   # rolling context window
    memory_summarize_after: int = 40
    session_name: str = "default"
    # voice identity (ChatGPT-style voices + speed)
    tts_voice: str = "en-US-AriaNeural"      # edge-tts voice name
    tts_speed: float = 1.0                   # 0.75 - 1.5 playback rate
    backchannel: bool = True                 # "mm-hmm?" acks (GPT-Live parity)
    # reasoning effort + escalation (GPT-Live delegation parity)
    reasoning_effort: str = "instant"        # instant | deep
    deep_llm_base_url: str = ""              # empty = same LLM
    deep_llm_model: str = ""                 # e.g. a bigger local model or gpt-4o
    # autonomy: "confirm" | "auto_safe" | "full"
    autonomy: str = "full"
    kill_switch_keys: str = "ctrl+alt+q"
    # ui
    dashboard_port: int = 8765
    # tool tuning
    move_duration_ms: int = 150
    type_interval_sec: float = 0.02
    # logging
    log_dir: str = str(ROOT / "logs")
    extra: dict = field(default_factory=dict)

    def apply_profile(self, name: str | None = None) -> dict:
        """Apply a PROFILES entry to this config; returns effective settings."""
        name = name or self.profile
        if name not in PROFILES:
            raise ValueError(f"unknown profile '{name}' "
                             f"(have: {', '.join(PROFILES)})")
        prof = PROFILES[name]
        self.profile = name
        self.asr_backend = prof["asr_backend"]
        self.llm_base_url = prof["llm_base_url"]
        self.llm_model = prof["llm_model"]
        self.tts_backend = prof["tts_backend"]
        return {"profile": name, **{k: v for k, v in prof.items()
                                    if k != "label"}}

    def api_key(self, env_name: str | None) -> Optional[str]:
        if not env_name:
            return None
        return os.environ.get(env_name)


_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def load_config(path: Path | None = None) -> Config:
    path = path or ROOT / "config.yaml"
    cfg = Config()
    if path.exists():
        try:
            import yaml
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
                else:
                    cfg.extra[k] = v
        except Exception as exc:  # noqa: BLE001
            import logging
            logging.getLogger("pai").warning("config load failed: %s", exc)
    cfg.apply_profile()   # resolve profile → concrete endpoints
    return cfg


def save_config(cfg: Config, path: Path | None = None) -> None:
    path = path or ROOT / "config.yaml"
    import yaml
    d = asdict(cfg)
    d.pop("extra", None)
    path.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")

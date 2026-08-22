"""Global configuration for PersonalAI-Assistant.

All knobs live here + an optional config.yaml the user can edit without
touching code. Provider selection is runtime-swappable.
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
# Runtime defaults
# ---------------------------------------------------------------------------


@dataclass
class Config:
    # "voicechat" | "modular" | "hybrid"
    provider: str = "voicechat"
    # modular pipeline pieces
    asr_backend: str = "whisper_server"     # whisper_server | faster_whisper
    llm_backend: str = "openai_compat"      # any OpenAI-compatible endpoint
    llm_base_url: str = "http://127.0.0.1:8081/v1"
    llm_model: str = "hy-mt2"
    tts_backend: str = "omnivoice"          # omnivoice | edge | voicechat
    # hybrid: vision handled by
    vlm_base_url: str = "http://127.0.0.1:8082/v1"
    vlm_model: str = "qwen2-vl"
    # voicechat provider
    voicechat_use_funchead: bool = True
    voicechat_port: int = 8123
    # audio
    sample_rate: int = 16000
    push_to_talk_seconds: float = 6.0
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
    return cfg


def save_config(cfg: Config, path: Path | None = None) -> None:
    path = path or ROOT / "config.yaml"
    import yaml
    path.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")

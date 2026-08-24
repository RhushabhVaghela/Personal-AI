"""Provider registry — swap brains at runtime."""
from __future__ import annotations

from pathlib import Path

from . import config


def get_provider(name: str | None = None):
    cfg = config.get_config()
    name = name or cfg.provider
    if name == "voicechat":
        from .provider_voicechat import VoiceChatProvider
        return VoiceChatProvider()
    if name == "modular":
        from .provider_modular import ModularProvider
        return ModularProvider()
    if name == "hybrid":
        from .provider_hybrid import HybridProvider
        return HybridProvider()
    if name == "qwen_omni":
        from .provider_qwen_omni import QwenOmniProvider
        return QwenOmniProvider()
    if name == "glm_voice":
        from .provider_glm_voice import GlmVoiceProvider
        return GlmVoiceProvider()
    if name == "moshi":
        from .provider_moshi import MoshiProvider
        return MoshiProvider()
    raise ValueError(f"unknown provider: {name}")


PROVIDERS = ["voicechat", "modular", "hybrid",
             "qwen_omni", "glm_voice", "moshi"]


# ---------------------------------------------------------------------------
# Capability matrix — drives dashboard control gating + setup warnings.
# A provider's built-in voice means the edge voice/speed/engine pickers are
# meaningless for it, so the UI disables them instead of silently ignoring.
# ---------------------------------------------------------------------------
def ui_caps(name: str) -> dict:
    """Which dashboard controls are meaningful for this provider."""
    cfg = config.get_config()
    if name == "modular":
        return {"voice": True, "speed": True, "engine": True, "effort": True}
    if name in ("voicechat", "hybrid"):
        # Nemotron voice is baked into the GGUF; edge re-voicing opt-in makes
        # the voice/speed pickers relevant
        edge = getattr(cfg, "voicechat_tts_mode", "native") == "edge"
        return {"voice": edge, "speed": edge, "engine": False, "effort": False}
    # qwen_omni (3 stock speakers), glm_voice (prompt-steered), moshi (fixed)
    return {"voice": False, "speed": False, "engine": False, "effort": False}


def missing_requirements(name: str) -> list[str]:
    """Human-readable setup items needed before this provider can run."""
    cfg = config.get_config()
    out: list[str] = []
    if name in ("voicechat", "hybrid"):
        if not config.VOICECHAT_EXE.exists():
            out.append(f"Nemotron exe missing: {config.VOICECHAT_EXE}")
        for gguf in (config.GGUF_MAIN, config.GGUF_MMPROJ, config.GGUF_TTS):
            if not gguf.exists():
                out.append(f"GGUF missing: {gguf.name}")
    if name == "qwen_omni":
        for mod, pip in (("torch", "torch (cu128 wheel)"),
                         ("transformers", "transformers>=4.50"),
                         ("soundfile", "soundfile")):
            try:
                __import__(mod)
            except ImportError:
                out.append(f"pip install {pip}")
        try:
            __import__("gptqmodel")
        except ImportError:
            out.append("pip install gptqmodel==2.0.0 (GPTQ kernel)")
    if name == "glm_voice":
        repo = Path(getattr(cfg, "glm_voice_repo", ""))
        if not repo.exists():
            out.append(f"git clone THUDM/GLM-4-Voice to {repo} (+conda env)")
    if name == "moshi":
        backend = getattr(cfg, "moshi_backend", "rust")
        repo = Path(getattr(cfg, "moshi_repo", ""))
        if backend == "wsl":
            out.append("WSL2 with CUDA + cargo (build inside /mnt/d/...)")
        else:
            exe = repo / "rust" / "target" / "release" / "moshi-backend.exe"
            if not exe.exists():
                out.append(
                    f"build moshi-backend.exe (Rust+CUDA nvcc): "
                    f"cd {repo}\\rust && cargo build --release --features cuda")
    return out

# ---------------------------------------------------------------------------
# Capability matrix — what each brain actually supports. Drives dashboard
# control disabling (server is source of truth; see check_available()).
# ---------------------------------------------------------------------------
PROVIDER_CAPS: dict[str, dict] = {
    # tts_engine: the 🔊 engine dropdown affects replies
    # voice:      the 🎙 voice picker affects replies
    # deep:       🧠 Deep mode (llm streaming) is meaningful
    "voicechat": {"tts_engine": False, "voice": False, "deep": False},
    "hybrid":    {"tts_engine": False, "voice": False, "deep": False},
    "qwen_omni": {"tts_engine": False, "voice": False, "deep": False},
    "glm_voice": {"tts_engine": False, "voice": False, "deep": False},
    "moshi":     {"tts_engine": False, "voice": False, "deep": False},
    "modular":   {"tts_engine": True,  "voice": True,  "deep": True},
}


def check_available(name: str) -> tuple[bool, str]:
    """Cheap pre-flight check — verifies prerequisites WITHOUT loading
    the model. Used before switching away from a working provider.

    Returns (available, reason_if_not).
    """
    cfg = config.get_config()

    if name == "modular":
        return True, ""          # degrades gracefully at turn time

    if name in ("voicechat", "hybrid"):
        needed = [config.VOICECHAT_EXE, config.GGUF_MAIN,
                  config.GGUF_MMPROJ, config.GGUF_TTS]
        missing = [str(p) for p in needed if not p.exists()]
        if missing:
            return False, "missing model files: " + ", ".join(missing)
        return True, ""

    if name == "qwen_omni":
        try:
            import torch, transformers, accelerate  # noqa: F401
            return True, ""
        except ImportError as exc:
            return False, (
                f"needs GPU torch + transformers ({exc}); install:\n"
                "  pip install torch --index-url "
                "https://download.pytorch.org/whl/cu128\n"
                "  pip install transformers accelerate soundfile")

    if name == "glm_voice":
        if not Path(cfg.glm_voice_repo).exists():
            return False, (
                f"GLM-4-Voice repo missing at {cfg.glm_voice_repo}. Clone:\n"
                "  git clone https://github.com/THUDM/GLM-4-Voice \""
                f"{cfg.glm_voice_repo}\"")
        return True, ""

    if name == "moshi":
        if cfg.moshi_backend == "wsl":
            return True, ""      # assume the WSL side is prepared
        exe = (Path(cfg.moshi_repo) / "rust" / "target" / "release" /
               "moshi-backend.exe")
        if not exe.exists():
            return False, (
                "MoshiVis backend not built yet. One-time build "
                "(needs Rust + CUDA nvcc):\n"
                f"  cd {cfg.moshi_repo}\\rust\n"
                "  cargo run --release --features cuda --bin moshi-backend"
                f" -- --config moshi-backend/config-{cfg.moshi_variant}.json"
                " standalone")
        return True, ""

    return False, f"unknown provider {name!r}"

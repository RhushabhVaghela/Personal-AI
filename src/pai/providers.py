"""Provider registry — swap brains at runtime."""
from __future__ import annotations

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

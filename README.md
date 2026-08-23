# PersonalAI-Assistant

> A local-first, realtime personal AI assistant that **listens, talks, sees your screen, and controls your PC** — endless hands-free conversation like ChatGPT Voice, but running on *your* hardware. Online providers available as an option; **offline is the default**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)](https://github.com/RhushabhVaghela/Personal-AI)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-active--development-orange)]()

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Hands-Free Mode (VAD)](#hands-free-mode-vad)
- [Memory — Hours-Long Conversations](#memory--hours-long-conversations)
- [Providers & Profiles](#providers--profiles)
- [Tools & Computer Control](#tools--computer-control)
- [Safety](#safety)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Hardware Requirements](#hardware-requirements)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

---

## Overview

PersonalAI-Assistant combines three capabilities normally spread across separate products:

1. **Realtime voice conversation** — always-on listening, it speaks when you speak, for hours
2. **Screen vision** — it captures and understands your display
3. **Computer control** — it clicks, types, scrolls, drags, and launches apps on your behalf

**Offline-first:** every core capability runs on your machine with zero cloud calls. If your hardware isn't up to it, flip a dropdown to route ASR/LLM/TTS through OpenAI or Groq instead — same UI, same tools, same memory.

Built and tested on an **RTX 5080 Laptop GPU (16 GB)**.

---

## Key Features

| Feature | Details |
|---|---|
| 👂 **Hands-free VAD** | Always-listening mic with end-of-speech detection — no buttons, just talk |
| 🔇 **Echo guard** | Mic gate closes while the assistant speaks so it never hears itself |
| 🗣️ **Speech-to-speech** | Nemotron VoiceChat 11B (GGUF Q4) — STT + reasoning + TTS in one model |
| 🧩 **Modular pipeline** | Any ASR → any OpenAI-compat LLM → any TTS, swappable at runtime |
| 💬 **Hours-long memory** | Persistent JSONL sessions + rolling context + auto-summarization |
| ☁️ **Online profiles** | `local` / `online-openai` / `online-groq` — one dropdown, env-var API keys |
| 👁️ **Screen vision** | Native mmproj perception or dedicated VLM (Qwen2-VL) |
| 🖱️ **Computer control** | 8 tools: screenshot, click, drag, scroll, type, keys, apps, shell |
| ⏹️ **Barge-in** | Stop button interrupts playback instantly; mic reopens |
| 🛑 **Kill-switch** | Global `Ctrl+Alt+Q` gates every input action mid-flight |
| ⚡ **Latency badge** | Per-turn response time shown live |

---

## Architecture

```
┌────────────────────────────── assistant core ──────────────────────────────┐
│                                                                            │
│  👂 mic ──► VAD engine ──► utterance wav ──┐      (echo-guarded)          │
│  ⌨️ text ──────────────────────────────────┤                               │
│                                            ▼                               │
│  Provider (runtime-swappable):                                             │
│    voicechat   Nemotron VoiceChat 11B GGUF (speech→speech, one model)      │
│    modular     ASR → LLM(+memory) → TTS   [local or online profile]        │
│    hybrid      VoiceChat voice + Qwen2-VL screen vision                    │
│                                            │                               │
│  💬 memory ◄── rolling context + summary ──┤                               │
│                                            ▼                               │
│  🧠 tool loop ──► JSON tool calls → ToolExecutor → results fed back        │
│                                                                            │
│  🛡️ safety ──► kill-switch hotkey + autonomy gating on EVERY action        │
└────────────────────────────────────────────────────────────────────────────┘
        │                                   │
   terminal (--hands-free)          web dashboard
                          mic ∙ chat ∙ audio queue ∙ stop ∙ live screenshot ∙
                          tool stream ∙ provider/profile pickers ∙ kill button
```

---

## Hands-Free Mode (VAD)

The flagship ChatGPT-Voice-style experience:

```
👂 ON → you talk → silence detected (~700 ms) → utterance sent →
assistant replies (mic gated) → playback ends → hangover → listening again
```

- **WebRTC VAD** (CPU-light, robust) with an adaptive **energy VAD** fallback
- Speech-start confirmed after ~90 ms of voice; end-of-speech after 700 ms silence
- **Echo guard**: while the reply plays, the mic gate is closed — no self-hearing loops
- Toggle in the dashboard (`👂 Hands-free`) or run `python -m pai.terminal --hands-free`

---

## Memory — Hours-Long Conversations

`ConversationStore` gives the assistant durable context:

- Every turn appended to `sessions/<name>.jsonl` — survives restarts
- Last N turns (`memory_turns: 24`) sent as rolling context every request
- Past that, older turns fold into a running **summary** kept at the front of the prompt (extractive by default; plug an LLM summarizer via `set_summarizer`)
- `reset()` archives the session and starts fresh

---

## Providers & Profiles

### Brains (what powers the conversation)

| Provider | Voice | Vision | Needs |
|---|---|---|---|
| `voicechat` | Nemotron VoiceChat 11B speech-to-speech | mmproj perception | nothing — self-launches |
| `modular` | ASR → LLM → TTS pipeline | — | backend services per profile |
| `hybrid` | VoiceChat | + Qwen2-VL | voicechat deps + VLM server |

### Profiles (where those services live)

| Profile | ASR | LLM | TTS | Offline? | Env key |
|---|---|---|---|---|---|
| `💻 local` *(default)* | Whisper server :9000 | any OpenAI-compat :8081 | OmniVoice :8889 / edge-tts | ✅ yes | — |
| `☁ online-openai` | Whisper API | GPT-4o-mini | OpenAI TTS | ❌ | `OPENAI_API_KEY` |
| `☁ online-groq` | whisper-large-v3 | Llama-3.3-70B | browser Web Speech | ❌ | `GROQ_API_KEY` |

Switch profiles from the dashboard dropdown — mid-session, no reload. Keys are read from environment variables only (never stored in config).

> **Groq note:** its free-tier API has no TTS endpoint, so replies are spoken by your browser's built-in Web Speech synthesis — zero extra latency, zero cost.

---

## Tools & Computer Control

| Tool | Parameters | Notes |
|---|---|---|
| `screenshot` | `monitor`, `max_width` | auto-downscaled for VLM payloads |
| `click` | `x, y, button?, double?` | animated move, validated buttons |
| `drag` | `x1, y1, x2, y2` | abortable mid-drag by kill-switch |
| `scroll` | `amount, x?, y?` | positive = up |
| `type_text` | `text` | literal typing |
| `press_key` | `key` | full pynput table: `f5`, `pgup`, `ctrl+s`, media keys… |
| `open_app` | `target` | name resolution + sanitized shell fallback |
| `run_command` | `command` | **blocked unless autonomy=full** |

Every execution is streamed to the dashboard's live tool panel.

---

## Safety

- **Kill-switch hotkey** `Ctrl+Alt+Q` — checked before *and during* every input action; drags abort mid-way; dashboard button too
- **Autonomy levels**: `confirm` (view-only) / `auto_safe` (no shell) / `full`
- **Process hygiene** — exactly one model process ever; switching providers unloads the old model first; Ctrl+C sweeps orphans and frees ports
- **Stale-port takeover** — a crashed instance's port is detected and released automatically

---

## Installation

```bash
cd D:\Agents-and-other-repos\PersonalAI-Assistant
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Requirements: Windows 10/11, Python 3.10+, FFmpeg on PATH.
Optional GPU models go in `D:\hoidhxd-NVIDIA-NemotronLabs-VoiceChat-11B-GGUF\` (path configurable in `src/pai/config.py`).

For the `local` profile's modular provider, also run: Whisper server (:9000), an OpenAI-compatible LLM (:8081), OmniVoice (:8889). For online profiles, set `OPENAI_API_KEY` / `GROQ_API_KEY`.

---

## Usage

```bash
# Web dashboard (recommended — full UX)
python -m pai.server            # → http://127.0.0.1:8765

# Terminal push-to-talk
python -m pai.terminal

# Terminal hands-free (just talk)
python -m pai.terminal --hands-free

# Options
python -m pai.terminal --provider modular --autonomy confirm --seconds 6
```

Dashboard controls: provider dropdown · profile dropdown (offline/online) ·
🎙 hold-to-talk · 👂 hands-free toggle · ⏹ Stop (barge-in) · text input ·
📸 screenshot · 🛑 kill-switch · live tool panel · latency badge.

---

## Configuration

`config.yaml` (all optional):

```yaml
provider: voicechat            # voicechat | modular | hybrid
profile: local                 # local | online-openai | online-groq

hands_free: true
vad_engine: auto               # auto | webrtc | energy
vad_silence_ms: 700            # end-of-speech detection
vad_aggressiveness: 2

memory_turns: 24               # rolling context window
memory_summarize_after: 40
session_name: default          # sessions/<name>.jsonl

autonomy: full                 # confirm | auto_safe | full
kill_switch_keys: ctrl+alt+q
dashboard_port: 8765
```

---

## Testing

```bash
python test_smoke.py         # 32 checks: capture, tools, autonomy, VAD-ready, providers
python test_integration.py   # dashboard HTTP + WS + live screenshot round-trip
```

All passing, plus live-verified: model boot/fallback, single-process guarantee,
graceful shutdown, stale takeover, hands-free toggles, profile switches,
end-to-end audio turns.

---

## Project Structure

```
src/pai/
  config.py              paths, PROFILES (local/online), VAD + memory settings
  server.py              web dashboard: HTTP+WS, shared provider lifecycle,
                         hands-free loop wiring, graceful shutdown
  terminal.py            push-to-talk + --hands-free CLI
  vad.py                 WebRTC/energy VAD state machines, echo guard,
                         continuous mic listener
  memory.py              ConversationStore: JSONL persistence, rolling
                         context, summarization
  providers.py           registry
  provider_voicechat.py  Nemotron speech-to-speech (file protocol)
  provider_modular.py    ASR→LLM(+memory)→TTS, local+online backends
  provider_hybrid.py     VoiceChat + VLM screen description
  screen_capture.py      MSS→PIL fallback, caching, downscaling
  input_control.py       pynput control + global KillSwitch
  tools.py               JSON tool protocol, autonomy matrix, event stream
  audio.py               mic record / playback
static/index.html        dashboard UI (dark, zero-build)
sessions/                persistent conversation JSONL files
```

---

## Hardware Requirements

| Component | Minimum | Tested |
|---|---|---|
| GPU (voicechat/hybrid) | 8 GB VRAM | RTX 5080 Laptop 16 GB |
| RAM | 16 GB | 32 GB |
| Disk | ~10 GB models | — |

No GPU? Use the `modular` provider with an **online profile**, or a small CPU
LLM via llama.cpp (`llm_base_url`). Everything else (VAD, capture, tools,
dashboard) is CPU-native.

---

## Roadmap

- [ ] Streaming/staged TTS (speak first sentence while generating rest)
- [ ] Wake word ("hey assistant") before VAD hand-off
- [x] ~~UIA element-tree clicking~~ → coordinates + screenshots work well today; element-name clicking next
- [ ] Wire screenshots into VoiceChat image turns natively
- [ ] Function-head GGUF support once exe builds support the architecture
- [ ] LLM-backed conversation summarizer hook-up
- [ ] System-tray app packaging

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `error 10048` port busy | Auto-handled — stale instance is killed; retry if persistent |
| Dashboard "connection closed" loop | Hard-refresh (`Ctrl+F5`) |
| `(timeout waiting for voicechat reply)` | First load ~10 s; check `nvidia-smi` for VRAM orphans |
| `unknown model architecture: voicechat_function_head` | Normal on older exe builds — auto-falls back |
| Mic never triggers in hands-free | Check default mic in Windows settings; try `vad_engine: energy` |
| Modular: connection refused :8081/:9000 | Start those services, or switch profile to online |
| numpy/PIL `ImportError` (wrong cp version) | `pip install --force-reinstall numpy pillow` in the project venv |

---

## Acknowledgements

Nemotron VoiceChat 11B (NVIDIA) · llama.cpp · Whisper · OmniVoice · Edge TTS ·
Qwen2-VL · webrtcvad · pynput · mss

---

## License

MIT — see [LICENSE](LICENSE).

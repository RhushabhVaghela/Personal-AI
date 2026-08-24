# PersonalAI-Assistant

> A local-first, realtime personal AI assistant that **listens, talks, sees your screen, and controls your PC** — endless hands-free conversation like ChatGPT Voice / Gemini Live, but running on *your* hardware. Online providers available as an option; **offline is the default**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)](https://github.com/RhushabhVaghela/Personal-AI)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-active--development-orange)]()

---

## Table of Contents

- [Overview](#overview)
- [Feature Matrix vs ChatGPT Voice & Gemini Live](#feature-matrix-vs-chatgpt-voice--gemini-live)
- [Architecture](#architecture)
- [Hands-Free Mode (VAD)](#hands-free-mode-vad)
- [Wake Word — "Proactive Audio"](#wake-word--proactive-audio)
- [Screen Sharing](#screen-sharing)
- [Memory — Hours-Long Conversations](#memory--hours-long-conversations)
- [Reminders & Proactive Speech](#reminders--proactive-speech)
- [Answer Cards](#answer-cards)
- [Voice Identity & Reasoning Effort](#voice-identity--reasoning-effort)
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

1. **Realtime voice conversation** — always-on listening with wake-word gating, for hours
2. **Screen vision** — on-demand screenshots or continuous Gemini-style screen sharing
3. **Computer control** — it clicks, types, scrolls, drags, and launches apps on your behalf *(something neither ChatGPT nor Gemini can do)*

**Offline-first:** every core capability runs locally with zero cloud calls. If your hardware isn't up to it, flip a dropdown to route ASR/LLM/TTS through OpenAI or Groq — same UI, same tools, same memory.

Built and tested on an **RTX 5080 Laptop GPU (16 GB)**.

---

## Feature Matrix vs ChatGPT Voice & Gemini Live

| Feature | ChatGPT Voice | Gemini Live | PersonalAI |
|---|---|---|---|
| Hands-free VAD listening | ✅ | ✅ | ✅ |
| Barge-in interrupt | ✅ full-duplex | ✅ | ✅ Stop button + mic reopen |
| Echo/self-hearing guard | ✅ | ✅ | ✅ |
| Wake word / responds only when addressed | ✅ | ✅ proactive audio | ✅ openWakeWord + transcript gate |
| Backchanneling ("mm-hmm") | ✅ GPT-Live | ✅ | ✅ cached ack clips |
| Captions overlay | ✅ cc button | ✅ auto when muted | ✅ every reply |
| Multiple voices | ✅ 9 | ✅ 10 | ✅ 7+ (any edge-tts voice) |
| Speaking speed control | ✅ by request | ✅ slider | ✅ slider 0.5–2× |
| Conversation memory across sessions | ✅ | ✅ | ✅ JSONL + rolling summary |
| Transcript history + export | ✅ in chat | ✅ in app | ✅ panel + JSONL export |
| Screen sharing | ✅ Advanced mode | ✅ | ✅ continuous stream + tool |
| Live video/camera | ✅ mobile | ✅ + overlays | ✅ webcam mode (OpenCV) |
| Streaming TTS (speak-while-generating) | ✅ | ✅ | ✅ sentence-chunked SSE |
| System-tray / background app | ✅ desktop app | ❌ | ✅ `python -m pai.tray` |
| App integrations (Calendar/Tasks) | limited | ✅ Google apps | ⚠️ reminders + Calendar mirror built-in |
| Reminders / proactive speech | via apps | ✅ | ✅ spoken aloud, offline |
| Answer cards (weather/time/widgets) | ✅ | ✅ | ✅ weather · clock · reminder |
| Reasoning effort selector | ✅ instant/deep | ✅ thinking modes | ✅ instant/deep w/ model escalation |
| Emotion-adaptive tone (affective) | ✅ | ✅ | ❌ needs better open S2S models |
| Background/locked-screen continuation | ✅ | ✅ hold/resume | ✅ tray app keeps server alive |
| **Real PC control** (click/type/apps/shell) | ❌ | ❌ | ✅ |
| Session length | ~limits | 15 min audio (!) | ♾️ unlimited |
| Fully offline | ❌ | ❌ | ✅ |

---

## Architecture

```
┌────────────────────────────── assistant core ──────────────────────────────┐
│                                                                            │
│  👂 mic ──► VAD engine ──► wake gate ──► utterance wav ─┐  (echo-guarded) │
│  ⌨️ text ────────────────────────────────────────────────┤                │
│                                                          ▼                │
│  Provider (runtime-swappable):                                           │
│    voicechat   Nemotron VoiceChat 11B GGUF (speech→speech, one model)    │
│    modular     ASR → LLM(+memory, tools) → TTS   [local or online]       │
│    hybrid      VoiceChat voice + Qwen2-VL screen vision                  │
│                                                          │                │
│  💬 memory ◄── rolling context + summary ────────────────┤                │
│  🖥 screen share ──► latest frame for look_at_screen ────┤                │
│                                                          ▼                │
│  🧠 tool loop ──► JSON tool calls → ToolExecutor → cards/events → back    │
│                                                                            │
│  🛡️ safety ──► kill-switch hotkey + autonomy gating on EVERY action       │
└────────────────────────────────────────────────────────────────────────────┘
        │                                   │
   terminal (--hands-free)          web dashboard
                      mic ∙ wake ∙ share-screen ∙ audio queue ∙ stop ∙ cards ∙
                      transcript ∙ voices/speed/effort pickers ∙ kill button
```

---

## Hands-Free Mode (VAD)

```
👂 ON → you talk → silence (~700 ms) → utterance sent →
assistant replies (mic gated) → playback ends → hangover → listening again
```

- **WebRTC VAD** (robust) with adaptive **energy VAD** fallback
- Speech-start confirmed after ~90 ms of voice; end-of-speech after 700 ms silence
- Toggle `👂 Hands-free` in the dashboard, or `python -m pai.terminal --hands-free`

## Wake Word — "Proactive Audio"

Arms hands-free so the assistant only responds when addressed:

```
📣 Armed → hears "hey assistant, what's the weather" → accepts ("what's the weather")
        → follow-ups stay active 90 s → re-arms
```

Two-layer hybrid:
1. **openWakeWord** (offline, CPU) — stock audio models: `hey_jarvis`, `alexa`, `hey_mycroft`, `hey_rhasspy`… configurable subset via `wake_oww_models`
2. **Transcript gate** (universal) — utterances without a configured phrase (`wake_phrases`) are dropped before reaching the model; works even with online providers

## Screen Sharing

Gemini-Live-style continuous sharing:

- `🖥 Share screen` toggles a change-detected capture loop (~0.7 s interval, 960 px frames) that streams live into the dashboard's screenshot panel
- The latest frame is kept server-side — the **`look_at_screen`** tool lets the assistant inspect what you're seeing on demand ("what am I looking at?")

## Memory — Hours-Long Conversations

- Every turn appended to `sessions/<name>.jsonl` (survives restarts)
- Last N turns (`memory_turns: 24`) sent as rolling context each request
- Older turns fold into a running **summary** kept at the front of the prompt
- Dashboard: `🆕 New chat` (archives), `🗒 Transcript` viewer, `⬇ Export`

## Reminders & Proactive Speech

```
You: "remind me at 5pm to call mom"     → set_reminder tool → card confirms
...at 5pm...                            → assistant SPEAKS UP unprompted 🔊
```

- Natural-time parsing: `at 5pm`, `at 17:30`, `in 10 minutes/hours` (past times roll to tomorrow)
- Persistent (`reminders.json`) · `list_reminders` / `cancel_reminder`
- Delivered as TTS to all connected dashboards, plus a ⏰ card

## Answer Cards

Floating widgets top-right of the dashboard (auto-expire ~20 s):

| Card | Trigger |
|---|---|
| 🌤 Weather (temp/desc/humidity, keyless wttr.in) | "what's the weather in Mumbai?" |
| 🕐 Clock (time + date) | "what time is it?" |
| ⏰ Reminder confirmation | any reminder set |

## Voice Identity & Reasoning Effort

- **Voice picker**: Aria, Jenny, Guy, Ana, Sonia (UK), Neerja/Prabhat (IN) — or any edge-tts voice name in config
- **Speed slider**: 0.5–2×, applied across edge-tts / OpenAI TTS
- **Backchannels**: short "Mm-hmm?" ack clip plays while the LLM thinks (`backchannel: false` to disable)
- **⚡ Instant / 🧠 Deep effort**: deep escalates to a bigger model via `deep_llm_base_url` / `deep_llm_model` (GPT-Live delegation parity)

---

## Providers & Profiles

### Brains

| Provider | Voice | Vision | Needs |
|---|---|---|---|
| `voicechat` | Nemotron VoiceChat 11B speech-to-speech | mmproj perception | nothing — self-launches |
| `modular` | ASR → LLM → TTS pipeline | — | backend services per profile |
| `hybrid` | VoiceChat | + Qwen2-VL | voicechat deps + VLM server |

### Profiles (where services live)

| Profile | ASR | LLM | TTS | Offline? | Env key |
|---|---|---|---|---|---|
| `💻 local` *(default)* | Whisper server :9000 | OpenAI-compat :8081 | OmniVoice :8889 / edge-tts | ✅ | — |
| `☁ online-openai` | Whisper API | gpt-4o-mini | OpenAI TTS | ❌ | `OPENAI_API_KEY` |
| `☁ online-groq` | whisper-large-v3 | llama-3.3-70B | browser Web Speech | ❌ | `GROQ_API_KEY` |

Switch profiles mid-session from the dashboard dropdown. Keys come from env vars only.

> **Groq note:** its free tier has no TTS endpoint, so replies use your browser's built-in Web Speech synthesis.

---

## Tools & Computer Control

| Tool | Parameters | Notes |
|---|---|---|
| `screenshot` | `monitor`, `max_width` | auto-downscaled payloads |
| `look_at_screen` | — | inspects the live shared frame |
| `set_reminder` | `when_text`, `text` | spoken aloud when due |
| `list_reminders` / `cancel_reminder` | `id?` | manage pending |
| `get_time` | — | emits clock card |
| `get_weather` | `location?` | wttr.in, emits weather card |
| `click` | `x, y, button?, double?` | animated move |
| `drag` | `x1, y1, x2, y2` | kill-switch abortable mid-drag |
| `scroll` | `amount, x?, y?` | positive = up |
| `type_text` | `text` | literal typing |
| `press_key` | `key` | full pynput table (`f5`, `pgup`, combos…) |
| `open_app` | `target` | name resolution + sanitized fallback |
| `run_command` | `command` | **blocked unless autonomy=full** |

All executions stream live into the dashboard's tool panel.

---

## Safety

- **Kill-switch hotkey** `Ctrl+Alt+Q` — checked before *and during* every input action; drags abort mid-way; dashboard button too
- **Autonomy levels**: `confirm` (view/info only) / `auto_safe` (input + reminders, no shell) / `full`
- **Process hygiene** — one model process ever; provider switch unloads old first; Ctrl+C sweeps orphans and frees ports; stale-port takeover on restart
- **Switch rollback + capability gating** — every provider is pre-flighted before a switch (model files, GPU deps, build state). Unavailable providers are rejected with setup instructions *without* touching your working model; if a start fails mid-switch the previous brain is automatically restored. The dashboard dims controls that don't apply to the selected brain (e.g. voice pickers on Nemotron, whose TTS voice is baked into the GGUF)

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
GPU models live in `D:\hoidhxd-NVIDIA-NemotronLabs-VoiceChat-11B-GGUF\` (paths in `src/pai/config.py`).

For `local` modular profile also run: Whisper server (:9000), OpenAI-compat LLM (:8081), OmniVoice (:8889).
For online profiles set `OPENAI_API_KEY` / `GROQ_API_KEY`.

First run of wake word downloads ONNX models (~10 MB) automatically.

---

## Usage

```bash
python -m pai.server                    # dashboard → http://127.0.0.1:8765
python -m pai.terminal                  # push-to-talk
python -m pai.terminal --hands-free     # always-listening
python -m pai.tray                      # system-tray app (server + icon)
```

Dashboard header: provider · profile · voice · 🔊 TTS engine · speed · effort · status · latency · hands-free badge · kill badge.
Composer: text input · 🎙 hold-to-talk · 👂 hands-free · 📣 wake word · 🖥 share screen · ⏹ stop · send.
Sidebar: 📸 screenshot · 🛑 kill-switch · 🆕 new chat · 🗒 transcript · ⬇ export · live tool panel.

---

## Configuration

`config.yaml` — see the fully-commented file in the repo root. Key sections:

```yaml
provider: voicechat            # voicechat | modular | hybrid |
                               # qwen_omni | glm_voice | moshi
profile: local                 # local | online-openai | online-groq

hands_free: true
wake_phrases: ["hey assistant", "assistant", "computer"]
wake_oww_models: ["hey_jarvis", "alexa"]
wake_custom_model: ""          # your trained .onnx (optional)

memory_turns: 24
session_name: default

tts_voice: en-US-AriaNeural
tts_speed: 1.0
tts_engine: edge               # edge | omnivoice | zipvoice | vibevoice | openai
tts_clone_ref: ""              # your voice wav → OmniVoice clones it
vibevoice_asr_url: ""          # optional realtime CPU ASR endpoint
voicechat_tts_mode: native     # native | edge (re-voice Nemotron replies)
backchannel: true

reasoning_effort: instant      # instant | deep
deep_model_hf: unsloth/Qwen3-30B-A3B-GGUF:UD-Q4_K_XL
deep_gpu_layers: 28            # attention on GPU, MoE experts in RAM
deep_idle_unload_s: 300        # auto-unload → VRAM returns to voice model

autonomy: full                 # confirm | auto_safe | full
kill_switch_keys: ctrl+alt+q
dashboard_port: 8765
```

---

## Testing

```bash
python test_smoke.py         # 35 checks
python test_rollback.py      # failed provider switch restores previous brain
python test_integration.py   # dashboard HTTP + WS round-trip
```

Live-verified end-to-end: model boot/fallback, single-process guarantee,
graceful shutdown, stale takeover, hands-free + wake gating, profile/voice/
effort switches, screen-share streaming, reminder scheduling + proactive
fire, weather/clock cards.

---

## Project Structure

```
src/pai/
  config.py              PROFILES (local/online), all runtime settings
  server.py              dashboard HTTP+WS, shared provider lifecycle,
                         hands-free + wake + share wiring, reminders,
                         graceful shutdown
  vad.py                 WebRTC/energy VAD, echo guard, mic listener
  wakeword.py            openWakeWord + transcript phrase gate (+ custom models)
  screenshare.py         change-detected continuous frame streamer
  webcam.py              OpenCV camera capture (camera-share mode)
  memory.py              ConversationStore: JSONL, rolling window, LLM summaries
  reminders.py           persistent store, natural-time parser, scheduler
  gcal.py                optional Google Calendar mirror for reminders
  stream_sanitizer.py    dual-channel FSM: strips <think>-style reasoning
                         from SSE streams (UI/TTS/tool-parser protection)
  provider_voicechat.py  Nemotron speech-to-speech (file protocol)
  provider_modular.py    ASR→LLM(+memory/tools)→TTS, local + online
  provider_hybrid.py     VoiceChat + VLM screen description
  provider_qwen_omni.py  Qwen2.5-Omni-7B GPTQ-Int4 thinker-talker S2S
  provider_glm_voice.py  GLM-4-Voice 9B int4 (emotion-steerable)
  provider_moshi.py      Moshi/MoshiVis full-duplex bridge (rust q8)
  deep_brain.py          on-demand Qwen3-30B-A3B MoE server (lazy, auto-unload)
  tts_engines.py         local neural TTS: OmniVoice / ZipVoice / VibeVoice
  tools.py               tool protocol, autonomy matrix, cards, events
  screen_capture.py      MSS→PIL fallback, cache, downscale
  input_control.py       pynput control + global KillSwitch
  audio.py               mic record / playback
static/index.html        dashboard UI
sessions/                conversation JSONL files
reminders.json           persisted reminders
```

## Choosing a voice brain

| You want | Pick |
|---|---|
| Best all-round offline S2S with tools | `qwen_omni` (GPTQ-Int4) |
| Natural chat, interrupt anytime | `moshi` (q8, full-duplex + vision) |
| Emotion/pace steered by voice command | `glm_voice` (int4) |
| Tool calling + reminders via prompt loop | `voicechat`, `modular`, or `hybrid` |
| No/weak GPU | `modular` + an online profile |

All fit in 16 GB VRAM one-at-a-time; the server unloads the previous model
before loading a new one.

---

## Hardware Requirements

| Component | Minimum | Tested |
|---|---|---|
| GPU (voicechat/hybrid) | 8 GB VRAM | RTX 5080 Laptop 16 GB |
| RAM | 16 GB | 32 GB |
| Disk | ~10 GB models | — |

No GPU? Use `modular` + an **online profile**, or a small CPU LLM. Everything
else (VAD, wake word, capture, tools, dashboard, reminders) is CPU-native.

---

## Roadmap

- [x] ~~Streaming/staged TTS~~ — SSE stream + sentence-chunked playback
- [x] ~~Custom wake-word models~~ — `wake_custom_model` loads your trained .onnx
- [x] ~~Google Calendar sync~~ — `google_calendar: true` mirrors reminders (gcal CLI or service account)
- [x] ~~Wire screenshots into VoiceChat image turns natively~~ — shared frame auto-attaches
- [x] ~~LLM-backed conversation summarizer hook-up~~ — wired with extractive fallback
- [x] ~~System-tray app packaging~~ — `python -m pai.tray`
- [x] ~~Webcam input alongside screen share~~ — camera mode via OpenCV
- [x] ~~Reasoning-tag stream sanitation~~ — dual-channel FSM (`stream_sanitizer.py`); <think> leakage can't reach UI/TTS/tool parser; reasoning shown in a collapsible panel
- [ ] Affective tone (emotion-adaptive prosody) — awaiting open S2S models
- [ ] Multi-user speaker identification

### Architecture notes: hardening against local-runtime failure modes

Three failure modes documented in local-agent deployments, and how this
project addresses them:

1. **Reasoning-tag leakage** (Qwen3/DeepSeek/GLM emit `<think>…</think>`;
   tags split across SSE chunks defeat string filters) → solved by the
   dual-channel FSM in `stream_sanitizer.py`: tag-open detection holds a
   sliding buffer, routes reasoning to a separate UI channel, guarantees
   clean text to the tool-call parser and TTS queue. For logit-level
   certainty, `deep_brain.chat(grammar=...)` accepts GBNF grammars that
   make malformed tool-call JSON structurally impossible.
2. **Context-cliff / memory saturation** → rolling window
   (`memory_turns`) + LLM summarization keeps prompts small; the Deep
   Brain runs llama.cpp with FlashAttention and q8_0 KV-cache quantization
   (linear KV scaling), and auto-unloads after idle so VRAM returns to the
   voice model. For multi-session persistence at scale, point
   `deep_llm_base_url`/`llm_base_url` at any server and pair it with an
   external memory layer (Mem0-style fact extraction is compatible with
   our JSONL session files).
3. **Quantized-model decoding instability** (broken JSON in tool calls) →
   mitigated two ways: sanitizer cleans the text channel before parsing,
   and GBNF grammar constraints are available for llama.cpp-served brains.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `error 10048` port busy | Auto-handled — stale instance killed; retry if persistent |
| Dashboard "connection closed" loop | Hard-refresh (`Ctrl+F5`) |
| `(timeout waiting for voicechat reply)` | First load ~10 s; check `nvidia-smi` for VRAM orphans |
| `unknown architecture: voicechat_function_head` | Normal on older exe builds — auto-falls back |
| Mic never triggers in hands-free | Check default mic; try `vad_engine: energy` |
| Modular: connection refused :8081/:9000 | Start those services or switch profile to online |
| numpy/PIL ImportError (wrong cp build) | `pip install --force-reinstall numpy pillow` in the venv |
| Wake word never fires | First run downloads ONNX models — check firewall; verify `openWakeWord loaded` in log |
| Webcam unavailable | `pip install opencv-python-headless`, check no other app holds the camera |
| Switching provider says "unavailable" | Read the message — it names the exact missing piece (build, repo, pip package) and your current model stays active |
| `<think>` text appears in replies | Shouldn't happen (FSM sanitizer); report it — or use a non-reasoning model for voicechat |
| Occasional MSS capture failure in logs | Transient screen-lock contention; PIL fallback engages automatically |

---

## Acknowledgements

Nemotron VoiceChat 11B (NVIDIA) · llama.cpp · Whisper · OmniVoice · Edge TTS ·
Qwen2-VL · openWakeWord · webrtcvad · pynput · mss · wttr.in

---

## License

MIT — see [LICENSE](LICENSE).

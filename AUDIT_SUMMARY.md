# Audit Summary — Architecture Framework Implementation
# (written to file to avoid chat truncation)

## What was implemented from the architecture doc

### 1. Reasoning-tag leakage → stream_sanitizer.py (NEW)
- Token-native FSM: TEXT → HOLD → THINK states, evaluated per SSE delta
- Sliding buffer matches 5 tag types: <think>, <thinking>, <thought>,
  <channel_thought>, <reasoning> — dynamic, no hardcoded regex updates
- Literal '<' in code/math passes through untouched
- Unclosed tags at EOS: synthetic closure, reasoning DISCARDED (never
  reaches answer text)
- Thinking routed to on_thinking callback -> new "thinking" WS channel ->
  collapsible panel in dashboard (auto-collapses when reply lands)
- Wired at TWO points: server streaming path (before TTS/UI) AND
  modular.think() before parse_tool_calls (CoT can't break JSON parsing)
- VERIFIED: split-tag chunks, literal <, unclosed, multi-tag, callback

### 2. Context cliff / memory saturation
- Already had: rolling window (memory_turns=24) + LLM summarizer +
  JSONL persistence (from earlier rounds)
- NEW in deep_brain.py: llama.cpp flags --flash-attn on (linear KV
  scaling), -ctk/-ctv q8_0 (quantized KV cache), auto-unload after idle
- README documents Mem0-style external memory compatibility

### 3. Quantized decoding instability (broken tool-call JSON)
- Sanitizer cleans text channel BEFORE parse_tool_calls
- deep_brain.chat(grammar=...) accepts GBNF grammars — logit-level
  constraint making invalid tool-call JSON structurally impossible
  (llama-server native support)

## Bug-fix audit (same session, full-file review) — 20 bugs fixed

CRITICAL:
1. glm_voice.stop() ran taskkill /F /IM python.exe — killed dashboard
2. set_effort: NameError 'cfg' undefined + missing global declaration;
   paused S2S model never restarted (now clears _active_provider properly)
3. moshi: asyncio.run from worker thread = fatal on py3.11+; now dedicated
   thread + own event loop; wss->ws (self-signed cert would always fail)

HIGH:
4. LLM summarizer dead code (_maybe_summarize never called it) — now used
5. sessions/ + reminders.json written OUTSIDE repo (parents[3]) — fixed
6. Streaming TTS ignored tts_engine config — local engines unwirable

MEDIUM:
7. think()/_build_messages double user-turn registration in memory
8. _tts_one hardcoded edge/omnivoice, ignored tts_engine
9. backchannel cache keyed by voice:speed but read one static file
10. _tts_input hardcoded Aria voice instead of cfg.tts_voice
11. qwen_omni tool follow-up sent SILENCE as audio — now injects via text
12. glm_voice tool results discarded instead of fed back into rounds

LOW:
13-20. duplicate tempfile_dir(), unused imports (Callable/Optional/
       collections), stale docstrings, reaper timer race, deep-brain
       status showing hardcoded model name, missing global decls

## Verification
35/35 smoke tests · integration PASS · all 13 modules parse+import clean
· sanitizer unit tests: split-tags/literal-</unclosed/multi/callback PASS

## Commits pushed
- stream_sanitizer FSM + wiring + dashboard thinking channel
- deep_brain FlashAttention/KV-quant/GBNF
- audit round 2 fixes (17 bugs)
- follow-up fixes (dedupe, tts_engine honor, backchannel cache)
- README architecture notes + config.yaml full documentation

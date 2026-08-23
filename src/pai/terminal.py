"""Terminal push-to-talk assistant loop.

Usage:
  python -m pai.terminal                 # default provider from config
  python -m pai.terminal --provider modular
  python -m pai.terminal --seconds 8
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import audio, config, input_control, providers, tools


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=providers.PROVIDERS, default=None)
    ap.add_argument("--seconds", type=float, default=None,
                    help="recording window (default: Enter-to-stop)")
    ap.add_argument("--autonomy", choices=["confirm", "auto_safe", "full"],
                    default=None)
    ap.add_argument("--hands-free", action="store_true",
                    help="always-listening VAD mode (no Enter needed)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = config.get_config()

    ks = input_control.get_kill_switch(cfg.kill_switch_keys)
    print("=" * 60)
    print(f" PersonalAI-Assistant | autonomy={cfg.autonomy}")
    print(f" KILL-SWITCH: {cfg.kill_switch_keys} (toggle anytime)")
    print("=" * 60)

    name = args.provider or cfg.provider
    if args.autonomy:
        cfg.autonomy = args.autonomy

    provider = providers.get_provider(name)
    if hasattr(provider, "start"):
        print(f"Loading {name} model (first load can take a minute)...")
        provider.start()
        if hasattr(provider, "is_running") and not provider.is_running():
            print("⚠ model failed to start — check the log above")
            return

    executor = tools.ToolExecutor(autonomy=cfg.autonomy)

    if args.hands_free:
        from .vad import get_engine, listen_continuous
        import threading
        engine = get_engine(sample_rate=cfg.sample_rate,
                            silence_hangover_ms=cfg.vad_silence_ms)
        stop = threading.Event()

        def on_utterance(wav_path: Path):
            print(f"\n🎙 (you) — processing {wav_path.name}...")
            engine.set_speaking(True)
            try:
                if provider.name == "modular":
                    text = provider.transcribe(wav_path)
                    print(f"🧑 You: {text}")
                    result = provider.think(text, executor)
                    print(f"🤖 Assistant: {result['text']}")
                    if result.get("text"):
                        out = Path(__file__).resolve().parents[2] / "logs" / "reply.wav"
                        spoken = provider.speak(result["text"], out)
                        audio.play_any(spoken)
                else:
                    result = provider.turn(wav_path, executor)
                    print(f"🤖 Assistant: {result.get('text') or '(voice reply)'}")
                    if result.get("audio"):
                        audio.play_any(Path(result["audio"]))
            except Exception as exc:  # noqa: BLE001
                print(f"(turn failed: {exc})")
            finally:
                engine.set_speaking(False)
            print("👂 listening...\n")

        print("👂 HANDS-FREE MODE — just talk; Ctrl+C to quit")
        try:
            listen_continuous(engine, on_utterance, stop_flag=stop)
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            if hasattr(provider, "stop"):
                provider.stop()
        return

    while True:
        try:
            cmd = input("\n[Enter]=record  [q]=quit > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd in ("q", "quit", "exit"):
            break
        if ks.engaged:
            print("⚠ Kill-switch engaged — press it again to release.")
            continue

        wav = (audio.record_seconds(args.seconds or cfg.push_to_talk_seconds)
               if args.seconds else audio.record_until_enter())
        if ks.engaged:
            print("⚠ Kill-switch engaged during recording; skipping turn.")
            continue

        try:
            if provider.name == "modular":
                transcript = provider.transcribe(wav)
                print(f"🧑 You: {transcript}")
                result = provider.think(transcript, executor)
                print(f"🤖 Assistant: {result['text']}")
                if result.get("text"):
                    out = Path(__file__).resolve().parents[2] / "logs" / "reply.wav"
                    out.parent.mkdir(exist_ok=True)
                    spoken = provider.speak(result["text"], out)
                    audio.play_any(spoken)
            else:
                result = provider.turn(wav, executor)
                text = result.get("text")
                if not text and result.get("audio"):
                    try:
                        text = "(voice reply — start whisper server for transcript)"
                        from .provider_modular import ModularProvider
                        text = ModularProvider().transcribe(
                            Path(result["audio"]))
                    except Exception:
                        pass
                print(f"🤖 Assistant: {text or '(no reply)'}")
                if result.get("audio"):
                    audio.play_any(Path(result["audio"]))
        except KeyboardInterrupt:
            print("\n(interrupted)")
        except Exception as exc:  # noqa: BLE001
            logging.getLogger("pai").exception("turn failed: %s", exc)

    if hasattr(provider, "stop"):
        provider.stop()
    print("Bye~ 👋")


if __name__ == "__main__":
    main()

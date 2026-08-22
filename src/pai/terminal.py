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
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = config.get_config()
    if args.provider:
        cfg.provider = args.provider
    if args.autonomy:
        cfg.autonomy = args.autonomy

    ks = input_control.get_kill_switch(cfg.kill_switch_keys.replace("+", " + "))
    print("=" * 60)
    print(f" PersonalAI-Assistant | provider={cfg.provider} | "
          f"autonomy={cfg.autonomy}")
    print(f" KILL-SWITCH: {cfg.kill_switch_keys} (toggle anytime)")
    print("=" * 60)

    provider = providers.get_provider(cfg.provider)
    if hasattr(provider, "start"):
        print("Loading model (first load can take a minute)...")
        provider.start()

    executor = tools.ToolExecutor(autonomy=cfg.autonomy)

    while True:
        try:
            cmd = input("\n[Enter]=record  [q]=quit > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd in ("q", "quit", "exit"):
            break
        if ks.engaged:
            print("⚠️ Kill-switch engaged — press it again to release.")
            continue

        wav = (audio.record_seconds(args.seconds or cfg.push_to_talk_seconds)
               if args.seconds or cfg.push_to_talk_seconds
               else audio.record_until_enter())
        if ks.engaged:
            print("⚠️ Kill-switch engaged during recording; skipping turn.")
            continue

        try:
            if cfg.provider == "modular":
                transcript = provider.transcribe(wav)
                print(f"🧑 You: {transcript}")
                result = provider.think(transcript, executor)
                print(f"🤖 Assistant: {result['text']}")
                if result.get("text"):
                    out = Path("logs/reply.wav")
                    out.parent.mkdir(exist_ok=True)
                    spoken = provider.speak(result["text"], out)
                    audio.play_any(spoken)
            else:
                result = provider.turn(wav, executor)
                print(f"🤖 Assistant: {result.get('text') or result.get('audio')}")
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

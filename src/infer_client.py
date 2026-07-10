"""
Client for the hosted OmniVoice server (model.py).

Usage:
    python infer_client.py \\
        --text "Hello, this is a test of zero-shot voice cloning." \\
        --ref-audio audios/reference_audios/saavi_vb.wav \\
        --ref-text "hello sir, i hope sab theek chal raha hoga, batayiye mein aapki kis tarah se madad kar sakti hun" \\
        --out audios/output_audios/saavi_out.wav
"""

from __future__ import annotations

import argparse
import base64

import requests


def generate(
    server_url: str,
    text: str,
    ref_audio: str | None = None,
    ref_text: str | None = None,
    language: str | None = None,
    instruct: str | None = None,
    speed: float | None = None,
    timeout: float = 120.0,
) -> bytes:
    """Calls POST /generate on a running model.py server and returns raw WAV bytes."""
    resp = requests.post(
        f"{server_url.rstrip('/')}/generate",
        json={
            "text": text,
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "language": language,
            "instruct": instruct,
            "speed": speed,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return base64.b64decode(data["audio_b64"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference against a hosted OmniVoice server")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--text", required=True)
    parser.add_argument("--ref-audio", default=None)
    parser.add_argument("--ref-text", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--instruct", default=None)
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--out", default="output.wav")
    args = parser.parse_args()

    wav_bytes = generate(
        args.server_url,
        text=args.text,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        language=args.language,
        instruct=args.instruct,
        speed=args.speed,
    )

    with open(args.out, "wb") as f:
        f.write(wav_bytes)
    print(f"Wrote {args.out} ({len(wav_bytes)} bytes)")


if __name__ == "__main__":
    main()

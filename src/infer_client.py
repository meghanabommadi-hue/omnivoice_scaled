"""
Client for the hosted OmniVoice server (model.py).

Usage:
    python infer_client.py \\
        --text "Hello, this is a test of zero-shot voice cloning." \\
        --ref-audio audios/reference_audios/saavi_vb.wav \\
        --ref-text "hello sir, i hope sab theek chal raha hoga, batayiye mein aapki kis tarah se madad kar sakti hun" \\
        --out audios/output_audios/saavi_out.wav

    # fire 20 concurrent requests at the already-running server and write
    # out/req0.wav .. out/req19.wav (does NOT load a second copy of the
    # model -- unlike tests/test_microbatch_server.py, this only talks
    # HTTP to whatever's already hosted by model.py)
    python infer_client.py \\
        --text "Hello, this is a test of zero-shot voice cloning." \\
        --ref-audio audios/reference_audios/saavi_vb.wav \\
        --ref-text "hello sir, i hope sab theek chal raha hoga, batayiye mein aapki kis tarah se madad kar sakti hun" \\
        --num-requests 20 --out-dir out

    # pseudo-streaming: split --text into 5-word chunks, synthesize them
    # in order, write each chunk's wav as soon as it's ready. OmniVoice
    # itself has no token-by-token streaming (see microbatch_server.py's
    # module docstring) -- this just approximates "first audio sooner" by
    # generating short chunks sequentially instead of the whole text at
    # once. Chunks are independent generate() calls, so expect a small
    # prosody/voice discontinuity at each chunk boundary.
    python infer_client.py \\
        --text "Hello, this is a test of zero-shot voice cloning across a longer sentence." \\
        --ref-audio audios/reference_audios/saavi_vb.wav \\
        --ref-text "hello sir, i hope sab theek chal raha hoga, batayiye mein aapki kis tarah se madad kar sakti hun" \\
        --stream-words 5 --out-dir stream_out
"""

from __future__ import annotations

import argparse
import base64
import os
import time
from concurrent.futures import ThreadPoolExecutor

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


def chunk_words(text: str, words_per_chunk: int) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i : i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference against a hosted OmniVoice server")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--text", required=True)
    parser.add_argument("--ref-audio", default=None)
    parser.add_argument("--ref-text", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--instruct", default=None)
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--out", default="output.wav",
                         help="Output path for a single request (ignored if --num-requests > 1).")
    parser.add_argument("-n", "--num-requests", type=int, default=1,
                         help="Fire this many concurrent requests at the server instead of one.")
    parser.add_argument("--out-dir", default="out",
                         help="Directory to write req<i>.wav files into when --num-requests > 1.")
    parser.add_argument("--concurrency", type=int, default=16,
                         help="Max requests in flight at once.")
    parser.add_argument("--stream-words", type=int, default=0,
                         help="Split --text into chunks of this many words and "
                              "synthesize them sequentially, writing each chunk's "
                              "wav as soon as it's ready (pseudo-streaming; "
                              "OmniVoice has no true token-by-token streaming). "
                              "Takes precedence over --num-requests.")
    args = parser.parse_args()

    if args.stream_words > 0:
        chunks = chunk_words(args.text, args.stream_words)
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"Streaming {len(chunks)} chunk(s) of ~{args.stream_words} words each "
              f"to {args.server_url} ...")
        wall_start = time.perf_counter()
        for i, chunk_text in enumerate(chunks):
            start = time.perf_counter()
            wav_bytes = generate(
                args.server_url,
                text=chunk_text,
                ref_audio=args.ref_audio,
                ref_text=args.ref_text,
                language=args.language,
                instruct=args.instruct,
                speed=args.speed,
            )
            elapsed = time.perf_counter() - start
            out_path = os.path.join(args.out_dir, f"chunk{i}.wav")
            with open(out_path, "wb") as f:
                f.write(wav_bytes)
            print(f"  chunk{i}.wav ready in {elapsed:.3f}s: \"{chunk_text}\"")
        wall_clock = time.perf_counter() - wall_start
        print(f"\nWrote {len(chunks)} chunk files to {args.out_dir}/ "
              f"in {wall_clock:.3f}s total (first chunk ready well before the rest)")
        return

    if args.num_requests <= 1:
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
        return

    os.makedirs(args.out_dir, exist_ok=True)

    def run_one(i: int) -> tuple[int, float]:
        start = time.perf_counter()
        wav_bytes = generate(
            args.server_url,
            text=args.text,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            language=args.language,
            instruct=args.instruct,
            speed=args.speed,
        )
        elapsed = time.perf_counter() - start
        out_path = os.path.join(args.out_dir, f"req{i}.wav")
        with open(out_path, "wb") as f:
            f.write(wav_bytes)
        return i, elapsed

    print(f"Firing {args.num_requests} concurrent requests at {args.server_url} "
          f"(concurrency={args.concurrency}) ...")
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for i, elapsed in pool.map(run_one, range(args.num_requests)):
            print(f"  req{i}.wav done in {elapsed:.3f}s")
    wall_clock = time.perf_counter() - wall_start

    print(f"\nWrote {args.num_requests} files to {args.out_dir}/ "
          f"in {wall_clock:.3f}s wall clock "
          f"({args.num_requests / wall_clock:.2f} req/s)")


if __name__ == "__main__":
    main()

"""
Load test for MicroBatchServer: fires >20 concurrent TTS requests and
reports per-request TTFT (== total latency, since generate() has no
partial-output streaming -- see microbatch_server.py docstring).

Usage:
    python test_microbatch_server.py -n 24
    python test_microbatch_server.py -n 40 --num-step 8 --max-batch-size 32
"""

import argparse
import asyncio
import os
import statistics
import sys
import time

import torch
from omnivoice import OmniVoice

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from microbatch_server import MicroBatchServer, TTSRequest

REF_AUDIO = "audios/reference_audios/saavi_vb.wav"
REF_TEXT = (
    "hello sir, i hope sab theek chal raha hoga, batayiye mein aapki kis "
    "tarah se madad kar sakti hun"
)

SAMPLE_SENTENCES = [
    "Namaste, aapki EMI is month due hai, please time par payment kar dijiye.",
    "Hello, this is a reminder call regarding your loan account overdue amount.",
    "Sir, aapke account mein do EMI installments pending hain, jaldi settlement kijiye.",
    "Good afternoon, please confirm a payment date to avoid penalty charges.",
    "Namaskar, aapka payment is hafte tak clear ho jana chahiye.",
    "Kripya apna outstanding balance jald se jald clear karein.",
    "This is a courtesy call to remind you about your upcoming due date.",
    "Aapke account par penalty lag sakti hai agar payment time par nahi hua.",
]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - f) * (values[c] - values[f])


async def main():
    parser = argparse.ArgumentParser(description="Load-test MicroBatchServer.")
    parser.add_argument("-n", "--num-requests", type=int, default=24)
    parser.add_argument("--language", type=str, default="hi")
    parser.add_argument("--num-step", type=int, default=16)
    parser.add_argument("--max-batch-size", type=int, default=24)
    parser.add_argument("--debounce-seconds", type=float, default=0.015)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    print(f"Loading OmniVoice on {args.device} ...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=args.device,
        dtype=torch.float16,
    )

    server = MicroBatchServer(
        model,
        debounce_seconds=args.debounce_seconds,
        max_batch_size=args.max_batch_size,
        num_step=args.num_step,
    )
    await server.start()

    def make_request(i: int) -> TTSRequest:
        return TTSRequest(
            text=SAMPLE_SENTENCES[i % len(SAMPLE_SENTENCES)],
            language=args.language,
            ref_audio=REF_AUDIO,
            ref_text=REF_TEXT,
        )

    if args.warmup:
        print(f"Warmup: {args.warmup} sequential request(s) (excluded from stats) ...")
        for i in range(args.warmup):
            await server.submit(make_request(i))

    print(f"Firing {args.num_requests} concurrent requests "
          f"(num_step={args.num_step}, max_batch_size={args.max_batch_size}, "
          f"debounce={args.debounce_seconds * 1000:.0f}ms) ...")

    wall_start = time.perf_counter()
    results = await asyncio.gather(
        *(server.submit(make_request(i)) for i in range(args.num_requests))
    )
    wall_clock = time.perf_counter() - wall_start

    await server.stop()

    ttfts = [r.total_s for r in results]
    batch_sizes = [r.batch_size for r in results]

    print("\n" + "=" * 60)
    print("MICROBATCH LOAD TEST SUMMARY")
    print("=" * 60)
    print(f"Requests:        {len(results)}")
    print(f"Wall clock:      {wall_clock:.3f}s")
    print(f"Throughput:      {len(results) / wall_clock:.2f} requests/s")
    print(f"Batch sizes seen: {sorted(set(batch_sizes))}")
    print("-" * 60)
    print("Per-request TTFT (submit -> audio ready; no partial streaming exists):")
    print(f"  mean:   {statistics.mean(ttfts):.3f}s")
    print(f"  median: {statistics.median(ttfts):.3f}s")
    print(f"  p90:    {percentile(ttfts, 90):.3f}s")
    print(f"  p99:    {percentile(ttfts, 99):.3f}s")
    print(f"  max:    {max(ttfts):.3f}s")
    under_1s = sum(1 for t in ttfts if t < 1.0)
    print(f"  under 1.0s: {under_1s}/{len(ttfts)}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

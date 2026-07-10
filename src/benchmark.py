"""
Benchmark OmniVoice generation over N requests.

Two modes:

- Batch mode (default): sends requests as one (or more) internally-batched
  `model.generate()` calls. `OmniVoice.generate()` is not a streaming API —
  it only returns once every item in the batch has finished — so TTFT here
  is a *proxy* equal to batch-completion latency (every request in a batch
  finishes at the same instant).

- Stream mode (`--stream`): splits every request's text into text chunks
  (see `split_text_stream_chunks`) and runs all N requests concurrently,
  batched by chunk position — round 1 batches every request's 1st chunk
  into a single `generate()` call, round 2 batches every request's 2nd
  chunk (for requests that still have one), and so on. This keeps requests
  batched (like real streaming TTS serving) while still surfacing a real
  per-request TTFT: the time from request start until round 1 (that
  request's first chunk) completes. Requests with fewer chunks simply drop
  out of later rounds.

  Because `generate()` only returns once the whole batch is done, TTFT
  under load necessarily grows with N (batching 16 requests' first chunks
  takes longer than batching 1). To make this visible, stream mode also
  runs a single-request baseline (TTFT-at-1, no concurrent load) before
  the N-request run, and reports both so you can see the latency floor
  alongside the throughput-oriented loaded number.

Text chunking rule (stream mode), scanned word by word:
    1. If a word ends with sentence-ending punctuation (./!/?), break there.
    2. Else if a word ends with a comma AND the chunk-so-far has >= 5 words,
       break there.
    3. Else if the chunk-so-far has reached 5 words, force a break.

Requests alternate language round-robin through `SAMPLE_SENTENCES` (currently
Hindi/Telugu) — every `generate()` call mixes languages in one batch, since
`language` is a per-item, not per-call, argument on the model.

Usage:
    python benchmark.py -n 32                       # 32 requests, one batch
    python benchmark.py -n 64 --batch-size 8         # 64 requests, 8 batches of 8
    python benchmark.py -n 16 --warmup 1
    python benchmark.py -n 16 --stream               # per-request streamed chunks, real TTFT
    python benchmark.py -n 24 --stream --words-per-chunk 6 --num-step 6   # tune for latency
    python benchmark.py -n 8 --save-audio out_wavs   # also write .wav files for listening
"""

import argparse
import os
import re
import statistics
import time
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
import torch
from omnivoice import OmniVoice

from batched_decode import enable_batched_decode

REF_AUDIO = "audios/reference_audios/anika_vb.mp3"
REF_TEXT = "जनता की सरकार जनता द्वारा जनता के लिए, पृथ्वी से नहीं मिटेगी"
# >1.0 tightens pacing / reduces inter-word gaps in generated speech; the
# Anika reference clip is a slow, deliberate reading, and that pacing style
# otherwise carries over into the cloned voice's output.
SPEED = 1.2

# Model default is num_step=32 with guidance_scale=2.0, t_shift=0.1. Each
# denoising step commits a chunk of audio tokens sized ~1/num_step of the
# total (see omnivoice.py's per-step `schedules`), so at low step counts
# (6-8) each step commits a much coarser chunk with fewer chances to correct
# a bad guess later — this is the direct cause of the quality drop.
# Compensate with a step-adaptive config: raise guidance_scale so each
# step's prediction is pulled harder toward the conditioned (voice-cloned)
# direction, and push t_shift toward 1.0 (uniform schedule) since the
# default's 0.1 front/back-loaded schedule was tuned assuming 32 steps of
# slack to recover in.
def generation_overrides_for(num_step: int) -> dict:
    if num_step >= 16:
        return {"guidance_scale": 2.0, "t_shift": 0.1}
    return {"guidance_scale": 3.5, "t_shift": 0.5}

# Alternating (language, text) pairs — cycled round-robin by `build_requests`
# so consecutive requests switch language, exercising the model's per-item
# (not per-call) language handling within a single batched `generate()` call.
SAMPLE_SENTENCES: list[tuple[str, str]] = [
    ("hi", "नमस्ते, आपकी EMI इस महीने due है, please time पर payment कर दीजिए।"),
    ("te", "నమస్కారం, మీ EMI ఈ నెల due అయ్యింది, దయచేసి వెంటనే payment చేయండి."),
    ("hi", "सर, आपके account में दो EMI installments pending हैं, जल्दी settlement कीजिए।"),
    ("te", "సర్, మీ account లో రెండు EMI installments pending ఉన్నాయి, త్వరగా settlement చేయండి."),
    ("hi", "नमस्कार, आपका payment इस हफ्ते तक clear हो जाना चाहिए।"),
    ("te", "నమస్కారం, మీ payment ఈ వారంలోపు clear అవ్వాలి."),
    ("hi", "कृपया अपना outstanding balance जल्द से जल्द clear करें।"),
    ("te", "దయచేసి మీ outstanding balance వీలైనంత త్వరగా clear చేయండి."),
    ("hi", "आपके account पर penalty लग सकती है अगर payment time पर नहीं हुआ।"),
    ("te", "మీ account మీద penalty పడే అవకాశం ఉంది, payment timeకి అవ్వకపోతే."),
    ("en", "Good afternoon, please confirm a payment date to avoid penalty charges."),
    ("te", "This is a courtesy call to remind you about your upcoming due date."),
]


@dataclass
class BatchResult:
    batch_index: int
    batch_size: int
    submit_time: float
    complete_time: float
    ttft: float = field(init=False)
    total_latency: float = field(init=False)

    def __post_init__(self):
        # No streaming signal available -> TTFT proxy == batch completion time.
        self.ttft = self.complete_time - self.submit_time
        self.total_latency = self.ttft


def build_requests(n: int) -> list[tuple[str, str]]:
    """Return `n` (language, text) pairs, cycling round-robin through
    `SAMPLE_SENTENCES` so consecutive requests alternate language."""
    return [SAMPLE_SENTENCES[i % len(SAMPLE_SENTENCES)] for i in range(n)]


def chunk(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


SENTENCE_END_RE = re.compile(r"[.!?]$")
COMMA_END_RE = re.compile(r",$")


def split_text_stream_chunks(text: str, words_per_chunk: int = 5) -> list[str]:
    """Split text into chunks to simulate streamed TTS input.

    Scans word by word and breaks a chunk when, in priority order:
      1. the current word ends with sentence-ending punctuation (./!/?)
      2. the current word ends with a comma AND the chunk-so-far already
         has >= `words_per_chunk` words
      3. the chunk-so-far has reached `words_per_chunk` words (hard cap,
         used when no punctuation triggers a break sooner)
    """
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []

    for word in words:
        current.append(word)
        if SENTENCE_END_RE.search(word):
            chunks.append(" ".join(current))
            current = []
        elif COMMA_END_RE.search(word) and len(current) >= words_per_chunk:
            chunks.append(" ".join(current))
            current = []
        elif len(current) >= words_per_chunk:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks


def crossfade_concat(chunks: list[np.ndarray], sample_rate: int, fade_ms: float = 20.0) -> np.ndarray:
    """Join consecutive streamed-chunk waveforms with a short equal-power
    crossfade instead of a hard concatenation.

    Each chunk is generated by an independent `generate()` call with no
    shared prosody state, so the raw sample at one chunk's end and the next
    chunk's start rarely line up — a plain `np.concatenate` can leave an
    audible click/discontinuity at every chunk boundary. Overlapping and
    blending the last `fade_ms` of one chunk with the first `fade_ms` of the
    next removes that boundary artifact without inserting silence (unlike
    `omnivoice.utils.audio.cross_fade_chunks`, which pads a silence gap
    between chunks — appropriate for the model's own long-form chunking, but
    not for reassembling one continuously-spoken sentence from stream
    chunks).
    """
    if len(chunks) == 1:
        return chunks[0]

    fade_n = int(fade_ms / 1000 * sample_rate)
    merged = chunks[0]

    for chunk in chunks[1:]:
        n = min(fade_n, merged.shape[-1], chunk.shape[-1])
        if n == 0:
            merged = np.concatenate([merged, chunk])
            continue

        # Equal-power crossfade so the blended region's perceived loudness
        # matches the un-faded audio on either side (linear fades dip in
        # the middle).
        t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
        fade_out, fade_in = np.cos(t), np.sin(t)

        head, overlap_tail = merged[:-n], merged[-n:]
        blended = overlap_tail * fade_out + chunk[:n] * fade_in
        merged = np.concatenate([head, blended, chunk[n:]])

    return merged


def run_batch(
    model: OmniVoice,
    requests: list[tuple[str, str]],
    batch_index: int,
    num_step: int,
    save_audio_dir: str | None = None,
) -> BatchResult:
    n = len(requests)
    languages, texts = zip(*requests)
    kwargs = dict(
        text=list(texts),
        language=list(languages),
        ref_text=[REF_TEXT] * n,
        ref_audio=[REF_AUDIO] * n,
        num_step=num_step,
        speed=SPEED,
        **generation_overrides_for(num_step),
    )

    submit_time = time.perf_counter()
    audios = model.generate(**kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    complete_time = time.perf_counter()

    if save_audio_dir and batch_index >= 0:  # skip warmup batches (negative index)
        for i, audio in enumerate(audios):
            path = os.path.join(save_audio_dir, f"batch{batch_index}_req{i}.wav")
            sf.write(path, audio, model.sampling_rate)

    return BatchResult(
        batch_index=batch_index,
        batch_size=n,
        submit_time=submit_time,
        complete_time=complete_time,
    )


@dataclass
class StreamResult:
    request_index: int
    num_chunks: int
    ttft: float
    total_latency: float


def run_stream_requests(
    model: OmniVoice,
    requests: list[tuple[str, str]],
    num_step: int,
    words_per_chunk: int = 5,
    save_audio_dir: str | None = None,
) -> list[StreamResult]:
    """Run all requests concurrently, batched by chunk position.

    Round 1 sends every request's 1st chunk in a single batched
    `generate()` call, round 2 sends every request's 2nd chunk (only for
    requests that still have one) in another batched call, and so on.
    This keeps the GPU fed with real batches (unlike per-request
    sequential streaming) while still measuring a genuine per-request TTFT:
    the elapsed time from the shared start until round 1 completes.

    Each request keeps its own language tag (`requests` is a list of
    (language, text) pairs), since `generate()` accepts a per-item language
    list — a single batched call can mix languages freely.
    """
    languages = [lang for lang, _ in requests]
    per_request_chunks = [split_text_stream_chunks(text, words_per_chunk) for _, text in requests]
    max_rounds = max(len(chunks) for chunks in per_request_chunks)

    ttft = [None] * len(requests)
    completion = [None] * len(requests)
    audio_chunks: list[list[np.ndarray]] = [[] for _ in requests]

    start_time = time.perf_counter()

    for round_idx in range(max_rounds):
        active_indices = [
            i for i, chunks in enumerate(per_request_chunks) if round_idx < len(chunks)
        ]
        if not active_indices:
            continue

        round_texts = [per_request_chunks[i][round_idx] for i in active_indices]
        round_languages = [languages[i] for i in active_indices]
        round_audios = model.generate(
            text=round_texts,
            language=round_languages,
            ref_text=[REF_TEXT] * len(active_indices),
            ref_audio=[REF_AUDIO] * len(active_indices),
            num_step=num_step,
            speed=SPEED,
            **generation_overrides_for(num_step),
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        round_time = time.perf_counter()

        for i, audio in zip(active_indices, round_audios):
            if ttft[i] is None:
                ttft[i] = round_time - start_time
            completion[i] = round_time - start_time
            if save_audio_dir:
                audio_chunks[i].append(audio)

    if save_audio_dir:
        for i, chunks in enumerate(audio_chunks):
            full_audio = crossfade_concat(chunks, model.sampling_rate)
            path = os.path.join(save_audio_dir, f"stream_req{i}.wav")
            sf.write(path, full_audio, model.sampling_rate)

    return [
        StreamResult(
            request_index=i,
            num_chunks=len(per_request_chunks[i]),
            ttft=ttft[i],
            total_latency=completion[i],
        )
        for i in range(len(requests))
    ]


def run_single_request_ttft(
    model: OmniVoice, request: tuple[str, str], num_step: int, words_per_chunk: int
) -> StreamResult:
    """Baseline TTFT-at-1: one request, no concurrent load, batch size 1 per
    chunk. This is the latency floor — what TTFT looks like with nothing
    else competing for the GPU, as opposed to the loaded (batch-of-N) TTFT
    from `run_stream_requests`, which necessarily grows with N since
    `generate()` only returns once the whole batch is done.
    """
    return run_stream_requests(model, [request], num_step, words_per_chunk)[0]


def summarize_stream(
    results: list[StreamResult], wall_clock: float, baseline_ttft: float | None = None
) -> None:
    ttfts = [r.ttft for r in results]
    totals = [r.total_latency for r in results]

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY (stream mode)")
    print("=" * 60)
    print(f"Requests:        {len(results)}")
    print(f"Wall clock:      {wall_clock:.3f}s")
    print(f"Throughput:      {len(results) / wall_clock:.2f} requests/s")
    if baseline_ttft is not None:
        print("-" * 60)
        print(f"TTFT-at-1 (baseline, no load):  {baseline_ttft:.3f}s")
        print(f"TTFT-at-{len(results)} (loaded, this run):   {statistics.mean(ttfts):.3f}s "
              f"({statistics.mean(ttfts) / baseline_ttft:.1f}x baseline)")
    print("-" * 60)
    print(f"Time to first chunk under load (TTFT-at-{len(results)}):")
    print(f"  mean:   {statistics.mean(ttfts):.3f}s")
    print(f"  median: {statistics.median(ttfts):.3f}s")
    print(f"  stdev:  {statistics.stdev(ttfts):.3f}s" if len(ttfts) > 1 else "  stdev:  n/a")
    print(f"  min:    {min(ttfts):.3f}s")
    print(f"  max:    {max(ttfts):.3f}s")
    print(f"  p50:    {percentile(ttfts, 50):.3f}s")
    print(f"  p90:    {percentile(ttfts, 90):.3f}s")
    print(f"  p99:    {percentile(ttfts, 99):.3f}s")
    print("-" * 60)
    print("Total request latency (all chunks):")
    print(f"  mean:   {statistics.mean(totals):.3f}s")
    print(f"  min:    {min(totals):.3f}s")
    print(f"  max:    {max(totals):.3f}s")
    print(f"  p90:    {percentile(totals, 90):.3f}s")
    print("=" * 60)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - f) * (values[c] - values[f])


def summarize(results: list[BatchResult], wall_clock: float) -> None:
    ttfts = [r.ttft for r in results]
    total_requests = sum(r.batch_size for r in results)

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Requests:        {total_requests}")
    print(f"Batches:         {len(results)}")
    print(f"Wall clock:      {wall_clock:.3f}s")
    print(f"Throughput:      {total_requests / wall_clock:.2f} requests/s")
    print("-" * 60)
    print("Per-batch latency / TTFT proxy (see caveat in module docstring):")
    print(f"  mean:   {statistics.mean(ttfts):.3f}s")
    print(f"  median: {statistics.median(ttfts):.3f}s")
    print(f"  stdev:  {statistics.stdev(ttfts):.3f}s" if len(ttfts) > 1 else "  stdev:  n/a")
    print(f"  min:    {min(ttfts):.3f}s")
    print(f"  max:    {max(ttfts):.3f}s")
    print(f"  p50:    {percentile(ttfts, 50):.3f}s")
    print(f"  p90:    {percentile(ttfts, 90):.3f}s")
    print(f"  p99:    {percentile(ttfts, 99):.3f}s")
    print("-" * 60)
    print("Per-request amortized latency within each batch:")
    per_req = [r.total_latency / r.batch_size for r in results]
    print(f"  mean:   {statistics.mean(per_req):.3f}s/request")
    print(f"  min:    {min(per_req):.3f}s/request")
    print(f"  max:    {max(per_req):.3f}s/request")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark OmniVoice generate() latency/throughput.")
    parser.add_argument("-n", "--num-requests", type=int, default=16, help="Total number of requests to send.")
    parser.add_argument("--batch-size", type=int, default=None, help="Requests per generate() call (default: all N in one batch).")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup batches/requests to run (excluded from stats).")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to load the model on.")
    parser.add_argument("--stream", action="store_true",
                         help="Split each request into text chunks (full stop / comma-after-N-words / "
                              "hard N-word cap) and run all requests concurrently, batched by chunk "
                              "position, measuring real per-request TTFT.")
    parser.add_argument("--words-per-chunk", type=int, default=5,
                         help="Chunk size for --stream mode's text splitting (comma/hard-cap threshold).")
    parser.add_argument("--num-step", type=int, default=32,
                         help="Iterative-unmasking steps passed to generate(). Lower = faster/lower "
                              "quality; model default is 32. This is the main lever for hitting a "
                              "sub-1s latency target.")
    parser.add_argument("--no-batched-decode", action="store_true",
                         help="Disable the batched-decode patch (batched_decode.py) and use the "
                              "stock per-sample codec decode loop instead.")
    default_audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audios", "output_audios")
    parser.add_argument("--save-audio", type=str, default=default_audio_dir,
                         help="Parent directory to write generated .wav files to (one per request; "
                              "not written for warmup runs or the stream-mode baseline). Each run "
                              "gets its own timestamped subdirectory under this path, named by mode "
                              "and --num-step, so different configs' outputs never overwrite each "
                              f"other. Default: {default_audio_dir}. Pass '' to skip saving audio.")
    args = parser.parse_args()

    if args.save_audio:
        mode = "stream" if args.stream else "batch"
        run_name = f"{mode}_step{args.num_step}_{time.strftime('%Y%m%d-%H%M%S')}"
        args.save_audio = os.path.join(args.save_audio, run_name)
        os.makedirs(args.save_audio, exist_ok=True)

    print(f"Loading OmniVoice on {args.device} ...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=args.device,
        dtype=torch.float16,
    )
    if not args.no_batched_decode:
        enable_batched_decode(model)

    all_requests = build_requests(args.num_requests)

    if args.stream:
        if args.warmup:
            print(f"Running {args.warmup} warmup round(s) (excluded from results) ...")
            warmup_requests = build_requests(args.batch_size or args.num_requests)
            for _ in range(args.warmup):
                run_stream_requests(model, warmup_requests, args.num_step, args.words_per_chunk)

        print("Running single-request baseline (TTFT-at-1, no concurrent load) ...")
        baseline = run_single_request_ttft(
            model, all_requests[0], args.num_step, args.words_per_chunk
        )
        print(f"  TTFT-at-1: {baseline.ttft:.3f}s (latency floor, unaffected by N)")

        print(f"Running {len(all_requests)} request(s) concurrently in stream mode "
              f"(num_step={args.num_step}, words_per_chunk={args.words_per_chunk}) ...")
        wall_start = time.perf_counter()
        results = run_stream_requests(
            model, all_requests, args.num_step, args.words_per_chunk,
            save_audio_dir=args.save_audio,
        )
        wall_clock = time.perf_counter() - wall_start
        if args.save_audio:
            print(f"Saved {len(results)} audio file(s) to {args.save_audio}/")

        for result in results:
            print(f"  request {result.request_index}: {result.num_chunks} chunks, "
                  f"ttft={result.ttft:.3f}s, total={result.total_latency:.3f}s")

        summarize_stream(results, wall_clock, baseline_ttft=baseline.ttft)
        return

    batch_size = args.batch_size or args.num_requests
    batches = chunk(all_requests, batch_size)

    if args.warmup:
        print(f"Running {args.warmup} warmup batch(es) (excluded from results) ...")
        warmup_requests = build_requests(batch_size)
        for i in range(args.warmup):
            run_batch(model, warmup_requests, batch_index=-1 - i, num_step=args.num_step)

    print(f"Running {len(batches)} batch(es) covering {args.num_requests} requests "
          f"(batch_size={batch_size}, num_step={args.num_step}) ...")

    results: list[BatchResult] = []
    wall_start = time.perf_counter()
    for i, requests in enumerate(batches):
        result = run_batch(
            model, requests, batch_index=i, num_step=args.num_step,
            save_audio_dir=args.save_audio,
        )
        results.append(result)
        print(f"  batch {i}: {result.batch_size} requests, "
              f"ttft/completion={result.ttft:.3f}s")
    wall_clock = time.perf_counter() - wall_start
    if args.save_audio:
        total_saved = sum(r.batch_size for r in results)
        print(f"Saved {total_saved} audio file(s) to {args.save_audio}/")

    summarize(results, wall_clock)


if __name__ == "__main__":
    main()

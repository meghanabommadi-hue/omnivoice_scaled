"""
Correctness + throughput check for the batched-decode patch.

Verifies that patching OmniVoice.generate() to batch the codec-decode step
(see batched_decode.py) produces numerically equivalent audio to the
original per-sample decode path, then reports the speedup.
"""

import os
import sys
import time

import numpy as np
import torch
from omnivoice import OmniVoice

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from batched_decode import enable_batched_decode

REF_AUDIO = "audios/reference_audios/saavi_vb.wav"
REF_TEXT = "hello sir, i hope sab theek chal raha hoga, batayiye mein aapki kis tarah se madad kar sakti hun"

TEXTS = [
    "Namaste, aapki EMI is month due hai, please time par payment kar dijiye.",
    "Hello, this is a reminder call regarding your loan account overdue amount.",
    "Sir, aapke account mein do EMI installments pending hain, jaldi settlement kijiye.",
    "Good afternoon, please confirm a payment date to avoid penalty charges.",
]


def main():
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cuda:0",
        dtype=torch.float16,
    )

    common_kwargs = dict(
        text=TEXTS,
        language=["hi"] * len(TEXTS),
        ref_audio=[REF_AUDIO] * len(TEXTS),
        ref_text=[REF_TEXT] * len(TEXTS),
    )

    torch.manual_seed(0)
    start = time.time()
    baseline_audios = model.generate(**common_kwargs)
    baseline_time = time.time() - start
    print(f"[baseline] per-sample decode: {baseline_time:.2f}s")

    enable_batched_decode(model)

    torch.manual_seed(0)
    start = time.time()
    batched_audios = model.generate(**common_kwargs)
    batched_time = time.time() - start
    print(f"[patched]  batched decode:    {batched_time:.2f}s")

    print(f"speedup: {baseline_time / batched_time:.2f}x")

    max_abs_diffs = []
    for i, (a, b) in enumerate(zip(baseline_audios, batched_audios)):
        n = min(len(a), len(b))
        diff = np.abs(a[:n] - b[:n])
        max_abs_diffs.append(diff.max())
        print(
            f"sample {i}: len_baseline={len(a)} len_batched={len(b)} "
            f"max_abs_diff={diff.max():.6f}"
        )

    overall_max_diff = max(max_abs_diffs)
    print(f"\noverall max abs diff: {overall_max_diff:.6f}")
    if overall_max_diff < 1e-2:
        print("PASS: batched decode is numerically equivalent to baseline.")
    else:
        print("WARNING: outputs differ more than expected — investigate.")


if __name__ == "__main__":
    main()

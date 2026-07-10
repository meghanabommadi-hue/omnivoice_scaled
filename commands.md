# Commands

All commands assume you're in the `omnivoice/` project root.

## Setup

```bash
./setup.sh
source .venv/bin/activate
```

## Host the model

Starts an HTTP server (FastAPI/uvicorn) that loads OmniVoice once and serves
`/generate` requests, using microbatching (`src/microbatch_server.py`) and the
batched codec-decode patch (`src/batched_decode.py`) under the hood.

```bash
python src/model.py --host 0.0.0.0 --port 8000
```

Env vars (optional):

| Var | Default | Purpose |
|---|---|---|
| `OMNIVOICE_MODEL_ID` | `k2-fsa/OmniVoice` | HF model id/path to load |
| `OMNIVOICE_DEVICE` | `cuda:0` | device_map passed to `from_pretrained` |
| `OMNIVOICE_NUM_STEP` | `16` | iterative-unmasking steps (lower = faster/lower quality) |
| `OMNIVOICE_MAX_BATCH_SIZE` | `24` | cap per `generate()` call |

Check it's up:

```bash
curl http://localhost:8000/health
```

## Run inference against the hosted model

```bash
python src/infer_client.py \
    --text "Hello, this is a test of zero-shot voice cloning." \
    --ref-audio audios/reference_audios/saavi_vb.wav \
    --ref-text "hello sir, i hope sab theek chal raha hoga, batayiye mein aapki kis tarah se madad kar sakti hun" \
    --out audios/output_audios/saavi_out.wav
```

Or with curl directly:

```bash
curl -s http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{
      "text": "Hello, this is a test of zero-shot voice cloning.",
      "ref_audio": "audios/reference_audios/saavi_vb.wav",
      "ref_text": "hello sir, i hope sab theek chal raha hoga, batayiye mein aapki kis tarah se madad kar sakti hun"
    }' | python -c "import sys,json,base64; d=json.load(sys.stdin); open('out.wav','wb').write(base64.b64decode(d['audio_b64']))"
```

## Run inference in-process (no server)

```bash
python src/inference.py
```

## Run the benchmark / load test scripts

```bash
python src/benchmark.py
python tests/test_microbatch_server.py -n 24
python tests/test_batched_decode.py
python tests/test_indian_languages_emi.py
```

## Layout

- `src/` — model hosting (`model.py`), client (`infer_client.py`), core modules
  (`microbatch_server.py`, `batched_decode.py`), and standalone scripts
  (`inference.py`, `inference_vllm_omni.py`, `benchmark.py`).
- `tests/` — correctness/load-test scripts.
- `audios/` — reference and sample/output audio.
- `results/` — benchmark comparison outputs (`stepN` batches of generated wavs).

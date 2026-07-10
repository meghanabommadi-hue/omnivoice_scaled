# NOTE: currently not runnable on this host. vllm-omni's OmniVoice pipeline
# (added in 0.20.0) requires a paired vllm version that hard-pins
# torch>=2.11 (CUDA 13 wheels). This driver (550.127.08, CUDA 12.9 max)
# cannot run CUDA 13 binaries. Needs a driver upgrade (or a CUDA-13-capable
# host) before vllm/vllm-omni can be installed and this script run.
import soundfile as sf
from vllm_omni import Omni

# Load the model (reads the bundled deploy/omnivoice.yaml diffusion-engine stage config)
model = Omni(model="k2-fsa/OmniVoice")

# Reference audio must be passed as a (samples, sample_rate) tuple
ref_audio, ref_sr = sf.read("audios/reference_audios/saavi_vb.wav", dtype="float32")

# Generate audio
outputs = model.generate(
    {
        "text": "Hello, this is a test of zero-shot voice cloning.",
        "ref_audio": (ref_audio, ref_sr),
        "ref_text": "hello sir, i hope sab theek chal raha hoga, batayiye mein aapki kis tarah se madad kar sakti hun",
    }
)

output = outputs[0]
audio = output.multimodal_output["audio"]
sample_rate = output.multimodal_output.get("audio_sample_rate", 24000)

sf.write("/home/jovyan/omnivoice/audios/output_audios/saavi_out.wav", audio, sample_rate)

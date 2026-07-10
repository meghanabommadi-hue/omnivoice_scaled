from omnivoice import OmniVoice
import soundfile as sf
import torch

# Load the model
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16
)

# Generate audio
audio = model.generate(
    text="Hello, this is a test of zero-shot voice cloning.",
    ref_audio="audios/reference_audios/saavi_vb.wav",
    ref_text="hello sir, i hope sab theek chal raha hoga, batayiye mein aapki kis tarah se madad kar sakti hun",
) # audio is a list of `np.ndarray` with shape (T,) at 24 kHz.

sf.write("audios/output_audios/saavi_out.wav", audio[0], 24000)

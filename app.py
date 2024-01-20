import argparse
import torch
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Transcribe a wav file.')
parser.add_argument('wavfile', type=str, help='The path to the wav file to transcribe.')
args = parser.parse_args()

MODEL_NAME = "openai/whisper-large-v3"
BATCH_SIZE = 8

device = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)

def transcribe(inputs, task):
    if inputs is None:
        raise Exception("No audio file submitted! Please upload or record an audio file before submitting your request.")

    text = pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)["text"]
    return  text

inputs = None
with open(args.wavfile, "rb") as f:
    inputs = f.read()
    inputs = ffmpeg_read(inputs, pipe.feature_extractor.sampling_rate)
    inputs = {"array": inputs, "sampling_rate": pipe.feature_extractor.sampling_rate}
    text = pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": "transcribe"}, return_timestamps=True)["text"]
    print(text)
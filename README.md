# Voice2Text CLI Utility

This is a simple command-line utility for converting voice to text. It uses the `openai/whisper-large-v3` model from Hugging Face's `transformers` library for automatic speech recognition.

## Installation

To install the necessary dependencies, run the following command:

```bash
python3 -m venv ./.venv
source ./.venv/bin/activate
pip install -r requirements.txt
```

Make sure all dependencies from `packages.txt` installed in your OS.

## Usage
You can run the script with the path to a .wav file as a command-line argument. For example:

```bash
python app.py path_to_your_file.wav
```

The script will print the transcription of the audio file to the console.

Using `--model` you can specify the model to fit your specific goals. For example:

```bash
python app.py path_to_your_file.wav --model=large
```
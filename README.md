# Voice2Text CLI Utility

This is a simple command-line utility for converting voice to text. It uses the `openai/whisper-large-v3` model from Hugging Face's `transformers` library for automatic speech recognition.

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage
You can run the script with the path to a .wav file as a command-line argument. For example:

```bash
python app.py path_to_your_file.wav
```

The script will print the transcription of the audio file to the console.

# psychopy-whisper

Speech-to-text transcription plugin for PsychoPy using [OpenAI Whisper](https://openai.com/research/whisper)

## Installing

Install this package with the following shell command:: 

    pip install psychopy-whisper

For GPU support, you will need to install additional packages via `pip` into the PsychoPy environment, see 
[here](https://pytorch.org/get-started/locally/) for more information and instructions. By default, the CPU version of 
PyTorch is installed and should work on most computers.

## Usage

Once the package is installed, PsychoPy will automatically load it when started and make objects available within the
`psychopy.sound.transcribe` namespace. You can select the backend to use for a session by specifying 
`'whisper'` when selecting a transcriber.
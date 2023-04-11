# psychopy-whisper

Speech-to-text transcription plugin for PsychoPy using [OpenAI Whisper](https://openai.com/research/whisper)

## Installing

Install this package with the following shell command:: 

    pip install psychopy-whisper

You may also use PsychoPy's builtin package manager to install this package.

## Usage

Once the package is installed, PsychoPy will automatically load it when started and make objects available within the
`psychopy.sound.transcribe` namespace. You can select the backend to use for a session by specifying 
`'whisper'` when selecting a transcriber.
"""
As this module is the value of an entry point targeting `psychopy.experiment.components.microphone`, 
any code below will be executed when `psychopy.plugins.activatePlugins` is called.
"""

from psychopy.experiment.components.microphone import MicrophoneComponent


# register whisper backend with MicrophoneComponent
if hasattr(MicrophoneComponent, "localTranscribers"):
    MicrophoneComponent.localTranscribers['OpenAI Whisper'] = "Whisper"
    if hasattr(MicrophoneComponent, "transcriberPaths"):
        MicrophoneComponent.transcriberPaths['Whisper'] = "psychopy_whisper.transcribe:WhisperTranscriber"

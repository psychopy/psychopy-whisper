"""
As this module is the value of an entry point targeting `psychopy.experiment.components.microphone`, 
any code below will be executed when `psychopy.plugins.activatePlugins` is called.
"""

from psychopy.experiment.components.microphone import MicrophoneComponent


# register whisper backend with MicrophoneComponent
if hasattr(MicrophoneComponent, "localTranscribers"):
    MicrophoneComponent.localTranscribers['OpenAI Whisper'] = "Whisper"
else:
    # if PsychoPy version predates this attribute (<2024.2.0), there should be a global instead
    from psychopy.experiment.components.microphone import localTranscribers, allTranscribers
    localTranscribers['OpenAI Whisper'] = "Whisper"
    allTranscribers['OpenAI Whisper'] = "Whisper"

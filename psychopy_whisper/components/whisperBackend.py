"""
As this module is the value of an entry point targeting `psychopy.experiment.components.microphone`, 
any code below will be executed when `psychopy.plugins.activatePlugins` is called.
"""

from psychopy.experiment.components.microphone import MicrophoneComponent


# register whisper backend with MicrophoneComponent
MicrophoneComponent.localTranscribers['OpenAI Whisper'] = "Whisper"

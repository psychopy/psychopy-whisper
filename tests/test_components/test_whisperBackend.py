def test_builder_detects_backend():
    """
    Check that PsychoPy Builder recognises the whisper backend as an option for MicrophoneComponent 
    once `activatePlugins` has been called.
    """
    # activate plugins
    from psychopy.plugins import activatePlugins
    activatePlugins()
    # import mic component
    from psychopy.experiment.components.microphone import MicrophoneComponent
    # check that whisper is in local transcribers list
    assert "OpenAI Whisper" in MicrophoneComponent.localTranscribers
    assert "Whisper" in list(MicrophoneComponent.localTranscribers.values())

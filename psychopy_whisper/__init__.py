#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Originally from the PsychoPy library
# Copyright (C) 2002-2018 Jonathan Peirce (C) 2019-2022 Open Science Tools Ltd.
# Distributed under the terms of the GNU General Public License (GPL).

"""Speech-to-text transcription using OpenAI Whisper.
"""

__version__ = '0.0.1'

from psychopy.sound.transcribe import TranscriptionResult, BaseTranscriber


def _download(url, root, in_memory):
    """Download a model for the OpenAI Whisper speech-to-text transcriber.

    This function is monkey-patched to override the `_download()` function to 
    use the `requests` library to get models from remote sources. This gets 
    around the SSL certificate errors we see with `urllib`.

    """
    # derived from source code found here:
    # https://github.com/openai/whisper/blob/main/whisper/__init__.py
    import hashlib
    import os
    import requests
    import warnings

    os.makedirs(root, exist_ok=True)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, os.path.basename(url))

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(
            f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        with open(download_target, "rb") as f:
            model_bytes = f.read()
        if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            return model_bytes if in_memory else download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not "
                f"match; re-downloading the file"
            )
    
    # here is the change we make that uses `requests` instead of `urllib`
    req = requests.get(url, allow_redirects=True)
    with open(download_target, 'wb') as dt:
        dt.write(req.content)

    model_bytes = open(download_target, "rb").read()
    if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not "
            "match. Please retry loading the model."
        )

    return model_bytes if in_memory else download_target


class WhisperTranscriber(BaseTranscriber):
    """Class for speech-to-text transcription using OpenAI Whisper.

    This class provides an interface for OpenAI Whisper for off-line (local) 
    speech-to-text transcription in various languages. You must first download
    the model used for transcription on the local machine.

    Parameters
    ----------
    initConfig : dict or None
        Options to configure the speech-to-text engine during initialization. 
    
    """
    _isLocal = True
    _engine = u'whisper'
    _longName = u"OpenAI Whisper"
    def __init__(self, initConfig=None):
        super(WhisperTranscriber, self).__init__(initConfig)

        # pull in imports
        import whisper
        whisper._download = _download  # patch download func using `requests`

        initConfig = {} if initConfig is None else initConfig

        self._device = initConfig.get('device', 'cpu')  # set the device
        self._modelName = initConfig.get('model_name', 'base')  # set the model

        # setup the model
        self._model = whisper.load_model(self._modelName).to(self._device)

    @property
    def device(self):
        """Device in use (`str`).
        """
        return self._device
    
    @property
    def model(self):
        """Model in use (`str`).
        """
        return self._modelName

    @staticmethod
    def downloadModel(modelName):
        """Download a model from the internet onto the local machine.
        
        Parameters
        ----------
        modelName : str
            Name of the model to download. You can get available models by 
            calling `getAllModels()`.

        Notes
        -----
        * In order to download models, you need to have the correct certificates
          installed on your system.
        
        """
        import whisper
        # todo - specify file name for the model
        whisper.load_model(modelName)  # calling this downloads the model

    @staticmethod
    def getAllModels():
        """Get available language models for the Whisper transcriber (`list`).

        Parameters
        ----------
        language : str
            Filter available models by specified language code.

        Returns
        -------
        list 
            List of available models.

        """
        import whisper

        return whisper.available_models()
    
    def transcribe(self, audioClip, modelConfig=None, decoderConfig=None):
        """Perform a speech-to-text transcription of a voice recording.

        Parameters
        ----------
        audioClip : AudioClip, ArrayLike
            Audio clip containing speech to transcribe (e.g., recorded from a
            microphone). Can be either an :class:`~psychopy.sound.AudioClip` 
            object or tuple where the first value is as a Nx1 or Nx2 array of 
            audio samples (`ndarray`) and the second the sample rate (`int`) in 
            Hertz (e.g., ``(samples, 48000)``).
        modelConfig : dict or None
            Configuration options for the model.
        decoderConfig : dict or None
            Configuration options for the decoder.

        Returns
        -------
        TranscriptionResult
            Transcription result instance.

        Notes
        -----
        * Audio is down-sampled to 16Khz prior to conversion which may add some
          overhead.

        """
        if isinstance(audioClip, AudioClip):  # use raw samples from mic
            samples = audioClip.samples
            sr = audioClip.sampleRateHz
        elif isinstance(audioClip, (list, tuple,)):
            samples, sr = audioClip

        # whisper requires data to be a flat `float32` array
        waveform = np.frombuffer(
            samples, samples.dtype).flatten().astype(np.float32)
        
        # resample if needed
        if sr != 16000:
            import librosa
            waveform = librosa.resample(
                waveform, 
                orig_sr=sr, 
                target_sr=16000)
        
        # pad and trim the data as required
        import whisper.audio as _audio
        waveform = _audio.pad_or_trim(waveform)

        modelConfig = {} if modelConfig is None else modelConfig
        decoderConfig = {} if decoderConfig is None else decoderConfig

        # our defaults
        language = "en" if self._modelName.endswith(".en") else None
        temperature = modelConfig.get('temperature', 0.0)
        word_timestamps = modelConfig.get('word_timestamps', True)

        # initiate the transcription
        result = self._model.transcribe(
            waveform, 
            language=language, 
            temperature=temperature, 
            word_timestamps=word_timestamps,
            **decoderConfig)
        
        transcribed = result.get('text', '')
        transcribed = transcribed.split(' ')  # split words 
        language = result.get('langauge', '')

        # create the response value
        toReturn = TranscriptionResult(
            words=transcribed,
            unknownValue=False,
            requestFailed=False,
            engine=self._engine,
            language=language)
        toReturn.response = str(result)  # provide raw JSON response

        self.lastResult = toReturn
        
        return toReturn


if __name__ == "__main__":
    pass

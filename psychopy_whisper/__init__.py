#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Originally from the PsychoPy library
# Copyright (C) 2002-2018 Jonathan Peirce (C) 2019-2022 Open Science Tools Ltd.
# Distributed under the terms of the GNU General Public License (GPL).

"""Speech-to-text transcription using OpenAI Whisper.
"""

__version__ = '0.0.1'  # plugin version

import os
import psychopy.logging as logging
from psychopy.preferences import prefs
from psychopy.sound.audioclip import AudioClip
from psychopy.sound.transcribe import TranscriptionResult, BaseTranscriber
import numpy as np


class WhisperTranscriber(BaseTranscriber):
    """Class for speech-to-text transcription using OpenAI Whisper.

    This class provides an interface for OpenAI Whisper for off-line (local) 
    speech-to-text transcription in various languages. You must first download
    the model used for transcription on the local machine.

    This interface uses `faster-whisper` for transcription, which is a faster 
    version of the original `whisper` library provided by OpenAI.

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

        initConfig = {} if initConfig is None else initConfig

        self._device = initConfig.get('device', 'auto')  # set the device
        self._modelName = initConfig.get('model_name', 'base.en')  # set the model
        self._computeType = initConfig.get('compute_type', 'int8')

        import faster_whisper
        WhisperTranscriber.downloadModel(self._modelName)

        # setup the model
        self._model = faster_whisper.WhisperModel(
            self._modelName, 
            device=self._device, 
            compute_type=self._computeType)

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
        import faster_whisper

        # check if `modelName` is valid
        if modelName not in WhisperTranscriber.getAllModels():
            raise ValueError(
                "Model `{}` is not available for download.".format(modelName))

        # set the directory for the model, create if it doesn't exist
        userModelDir = os.path.join(
            prefs.paths['userPrefsDir'], 'cache', 'whisper')
        if not os.path.isdir(userModelDir):
            logging.info(
                "Creating directory for Whisper models: {}".format(
                    userModelDir))
            os.mkdir(userModelDir)

        # download the model
        faster_whisper.download_model(
            modelName, 
            output_dir=userModelDir,
            local_files_only=False  # avoid downloading if cached
        )

    @staticmethod
    def getAllModels():
        """Get available language models for the Whisper transcriber (`list`).

        Parameters
        ----------
        language : str
            Filter available models by specified language code.

        Returns
        -------
        tuple 
            Sequence of available models.

        """
        from faster_whisper import utils

        return utils._MODELS
    
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
        if int(sr) != 16000:
            import librosa
            waveform = librosa.resample(
                waveform, 
                orig_sr=sr, 
                target_sr=16000)
        
        # pad and trim the data as required
        # import faster_whisper.audio as _audio
        # waveform = _audio.pad_or_trim(waveform)

        modelConfig = {} if modelConfig is None else modelConfig
        decoderConfig = {} if decoderConfig is None else decoderConfig

        # our defaults
        language = "en" if self._modelName.endswith(".en") else None
        temperature = modelConfig.get('temperature', 0.0)
        word_timestamps = modelConfig.get('word_timestamps', True)

        # initiate the transcription
        segments, _ = self._model.transcribe(
            waveform, 
            language=language, 
            temperature=temperature, 
            word_timestamps=word_timestamps,
            **decoderConfig)
        results = list(segments)

        # create the response value

        return results
        
        # transcribed = result.get('text', '')
        # transcribed = transcribed.split(' ')  # split words 
        # language = result.get('langauge', '')

        # # create the response value
        # toReturn = TranscriptionResult(
        #     words=transcribed,
        #     unknownValue=False,
        #     requestFailed=False,
        #     engine=self._engine,
        #     language=language)
        # toReturn.response = str(result)  # provide raw JSON response

        # self.lastResult = toReturn
        
        # return toReturn


if __name__ == "__main__":
    print(WhisperTranscriber.getAllModels())
    transc = WhisperTranscriber({'model_name': 'tiny.en'})

    # test transcription

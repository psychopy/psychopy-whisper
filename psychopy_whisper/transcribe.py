import os
import json
import psychopy.logging as logging
import numpy as np
import time
from psychopy.preferences import prefs
from psychopy.sound.audioclip import AudioClip
from psychopy.sound.transcribe import TranscriptionResult, BaseTranscriber


class WhisperTranscriber(BaseTranscriber):
    """Class for speech-to-text transcription using OpenAI Whisper.

    This class provides an interface for OpenAI Whisper for offline (local) 
    speech-to-text transcription in various languages.     
    
    You must first download the model used for transcription on the local 
    machine. This can be done by calling `downloadModel(modelName)`. The
    available models can be obtained by calling `getAllModels()`.

    Transcription is computationally intensive and requires significant system 
    memory and processing power. For optimal performance, it is recommended to 
    enable GPU acceleration if available. By default, the transcriber will use
    the GPU if available, otherwise it will use the CPU. You can specify the
    device to use by setting the `device` option in the initialization
    configuration. In order to use GPU acceleration, you must have a compatible 
    GPU and the necessary drivers and packages installed.

    Parameters
    ----------
    initConfig : dict or None
        Options to configure the speech-to-text engine during initialization. 
        If `None` default values are used. The following options are available:
        `{'device': "auto", 'model_name': "tiny.en"}`. Values for `'device'`
        can be either `'cpu'`, `'gpu'` or `'auto'`, and `'model_name'` can be
        any value returned by `.getAllModels()`.
    
    """
    _isLocal = True
    _engine = u'whisper'
    _longName = u"OpenAI Whisper"
    def __init__(self, initConfig=None):
        super(WhisperTranscriber, self).__init__(initConfig)

        initConfig = {} if initConfig is None else initConfig

        self.userModelDir = None  # directory for user models
        self._device = initConfig.get('device', 'auto')  # set the device
        self._modelName = initConfig.get('model_name', 'tiny.en')  # set the model

        import torch
        import whisper

        if self._device == 'auto':
            # prefer GPU if available
            logging.info(
                "Decoder device set to 'auto', checking for GPU availability.")
            self._device = torch.cuda.is_available() and 'cuda' or 'cpu'

            # inform the user of the choice
            if self._device == 'cuda':
                logging.info("Using CUDA enabled GPU for Whisper transcriber.")
            else:
                logging.info(
                    "GPU support is not available on this system. Using CPU "
                    "for Whisper transcriber.")

        # check if `modelName` is valid
        if self._modelName not in WhisperTranscriber.getAllModels():
            raise ValueError(
                "Model `{}` is not available for download.".format(
                self._modelName))

        # set the directory for the model, create if it doesn't exist
        userModelDir = os.path.join(
            prefs.paths['userPrefsDir'], 'cache', 'whisper')
        if not os.path.isdir(userModelDir):
            logging.info(
                "Creating directory for Whisper models: {}".format(
                    userModelDir))
            os.mkdir(userModelDir)

        self.userModelDir = userModelDir

        # setup the model
        logging.info("Loading Whisper model: {}".format(self._modelName))

        tStartLoad = time.time()  # profiling
        self._model = whisper.load_model(
            self._modelName, 
            device=self._device, 
            # compute_type=self._computeType,
            download_root=self.userModelDir,  # downloads
            in_memory=initConfig.get('in_memory', True))
        tEndLoad = time.time()

        logging.debug(
            "Whisper model loaded in {:.2f} seconds.".format(
                tEndLoad - tStartLoad))

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

        If the model is already downloaded, the function will return the path to
        the model without downloading it again.
        
        Parameters
        ----------
        modelName : str
            Name of the model to download. You can get available models by 
            calling `getAllModels()`.

        Returns
        -------
        str
            Path to the dowloaded model.

        Notes
        -----
        * In order to download models, you need to have the correct certificates
          installed on your system.
        
        """
        import whisper

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

        # get the URL for the model
        modelURL = whisper._MODELS[modelName]  # get the model URL

        # download the model
        result = whisper._download(modelURL, userModelDir, in_memory=False)

        return result

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
        import whisper

        return whisper.available_models()
    
    def transcribe(self, audioClip, language=None, expectedWords=None, 
                   config=None, modelConfig=None, decoderConfig=None):
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
            # check if the audio clip needs to be resampled
            if audioClip.sampleRateHz != 16_000:
                logging.warning(
                    "Audio clip sample rate is not 16KHz. Resampling to 16KHz.")
                audioClip = audioClip.resample(16_000)

            samples = audioClip.samples
            sr = audioClip.sampleRateHz
        elif isinstance(audioClip, (list, tuple,)):
            samples, sr = audioClip
            if sr != 16_000:
                logging.warning(
                    "Audio clip sample rate is not 16KHz. Resampling to 16KHz.")
                samples = AudioClip(samples, sr).resample(16_000).samples
                sr = 16_000
        else:
            raise ValueError(
                "Invalid audio clip provided. Must be an AudioClip or a tuple "
                "containing audio samples and sample rate.")
            
        # remove excess channels for recording if not mono audio
        if samples.ndim > 1:
            logging.warning(
                "Audio clip has more than one channel. Only the first channel "
                "will be used for transcription.")
            samples = samples[:, 0]

        # format array as contiguous float32
        waveform = np.ascontiguousarray(samples, dtype=np.float32)
        duration = len(waveform) / sr

        # normalize the waveform between -1 and 1
        if np.max(np.abs(waveform)) > 1.0:
            logging.warning(
                "Audio clip has values outside the range [-1, 1]. Normalizing.")
            waveform /= np.max(np.abs(waveform))

        # pad and trim the data as required
        if modelConfig is None:
            if config is not None:
                modelConfig = config
            else:
                modelConfig = {}
        decoderConfig = {} if decoderConfig is None else decoderConfig

        # our defaults
        language = "en" if self._modelName.endswith(".en") else None

        # TODO - these should be set by `decoderConfig`, we'll correct this later
        temperature = modelConfig.get('temperature', 0.0)
        # timestamps, want this `True` in most cases
        word_timestamps = modelConfig.get('word_timestamps', True)  

        import whisper

        logging.debug(
            "Starting transcription for {:.2f} seconds of audio.".format(
                duration))

        # initiate the transcription
        tStart = time.time()  # profiling
        result = whisper.transcribe(
            self._model,
            waveform, 
            language=language, 
            temperature=temperature, 
            word_timestamps=word_timestamps,
            **decoderConfig)
        tEnd = time.time()

        logging.debug(
            "Transcription completed in {:.2f} seconds.".format(tEnd - tStart))

        # process the result from the transcriber
        segments = list(result['segments'])  # expand generator

        # compile words in all segments
        text = []
        for segment in segments:
            text.append(segment['text'])

        # create a JSON string from the segments
        dataStruct = {'segments': {}}  # initialize the data structure
        for segment in segments:
            wordDicts = {}
            for i, word in enumerate(segment['words']):
                wordDicts[i] = {
                    'word': word['word'].strip(),  # clear whitespace
                    'start': word['start'],
                    'end': word['end'],
                    'probability': word['probability']
                }
            dataStruct['segments'][segment['id']] = {
                'id': segment['id'],  # enum for segment
                'seek': segment['seek'],  # pos in file
                'start': segment['start'],
                'end': segment['end'],
                'text': (segment['text']).strip(),
                'tokens': segment['tokens'],
                'temperature': segment['temperature'],
                'avg_logprob': segment['avg_logprob'],
                'compression_ratio': segment['compression_ratio'],
                'no_speech_prob': segment['no_speech_prob'],
                'words': wordDicts
            }

        text = ' '.join(text).strip()  # remove excess whitespace
        transcribed = text.split(' ')  # split words 

        # create the response value
        toReturn = TranscriptionResult(
            words=transcribed,
            unknownValue=False,
            requestFailed=False,
            engine=self._engine,
            language=language)
        toReturn.response = json.dumps(
            dataStruct, indent=4)  # provide raw response

        self.lastResult = toReturn
        
        return toReturn
    

_whisperTranscriber = None  # instance for the transcriber

def recognizeWhisper(audioClip=None, language=None, expectedWords=None,
                     config=None):
    """Perform speech-to-text conversion on the provided audio samples locally 
    using OpenAI Whisper.

    This is a self-hosted (local) transcription engine. This means that the 
    actual transcription is done on the host machine, without passing data over 
    the network. 

    Parameters
    ----------
    audioClip : :class:`~psychopy.sound.AudioClip` or None
        Audio clip containing speech to transcribe (e.g., recorded from a
        microphone). Specify `None` to initialize a client without performing a
        transcription, this will reduce latency when the transcriber is invoked
        in successive calls. Any arguments passed to `config` will be sent to
        the initialization function of the model if `None`.
    language : str or None
        Language code (eg., 'en'). Unused for Whisper since models are 
        multi-lingual, but may be used in the future. 
    expectedWords : list or None
        Not required by Whisper but kept here for compatibility.
    config : dict or None
        Additional configuration options for the specified engine. For Whisper,
        the following configuration dictionary can be used 
        `{'device': "auto", 'model_name': "base"}`. Values for `'device'` can be 
        either `'cpu'`, `'gpu'` or `'auto'`, and `'model_name'` can be any value 
        returned by `.getAllModels()`. 

    Returns
    -------
    TranscriptionResult
        Transcription result object.

    """
    global _whisperTranscriber

    config = {} if config is None else config
    onlyInitialize = audioClip is None
    isInitialized = _whisperTranscriber is not None

    if not isInitialized:
        # initialization options
        initConfig = {
            'device': config.get('device', 'auto'), 
            'model_name': config.get('model_name', 'tiny.en')
            # 'compute_type': config.get('compute_type', 'int8')
        }
        _whisperTranscriber = WhisperTranscriber(initConfig)

    if onlyInitialize:  # only initialize
        return None 
    
    # set parameters which we used to support
    config['expectedWords'] = expectedWords
    config['language'] = language
    
    # do transcription and return result
    return _whisperTranscriber.transcribe(audioClip, modelConfig=config)


if __name__ == "__main__":
    pass

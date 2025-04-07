#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Originally from the PsychoPy library
# Copyright (C) 2002-2018 Jonathan Peirce (C) 2019-2022 Open Science Tools Ltd.
# Distributed under the terms of the GNU General Public License (GPL).

"""Speech-to-text transcription using OpenAI Whisper.
"""

__all__ = [
    "WhisperTranscriber",
    "recognizeWhisper"
]

from .transcribe import WhisperTranscriber, recognizeWhisper

__version__ = '0.0.7'  # plugin version


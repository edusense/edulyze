"""
Author: Anonymized
Created: Anonymized

This file restructures audio input data in format required by analytics engine
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Import project level functions and classes
from configs.constants import Constants
from utils.time_utils import time_diff


def preprocess_audio_data(session_input_object, raw_audio_data, logger_pass):
    """
    Fetches raw audio information for teacher, student channel from graphql backend

    Parameters:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        raw_audio_data(dict)           : Dictionary containing audio data in raw graphql format
        logger_pass(logger)            : The parent logger from which to derive the logger for the engine

    Returns:
        processed_audio_data(dict)      : Dictionary containing audio data processed for analytics engine
    """
    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('preprocess_audio_data')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_process_audio_data_start = datetime.now()

    # initialize processed audio data
    processed_audio_data = dict({
        'student': {},
        'instructor': {}
    })

    audio_start_timestamp = int(
            raw_audio_data['student']['sessions'][0]['audioFrames'][0]['timestamp']['unixSeconds'])

    # Loop over each frame and reformat frame data for student side camera

    audioFrames = raw_audio_data['student']['sessions'][0]['audioFrames']

    for audioFrame in audioFrames:
        processed_audio_frame = preprocess_audio_frame(audioFrame, logger)
        processed_audio_frame['block_id'] = int(((processed_audio_frame['timestamp'] / Constants.MILLISECS_IN_SEC)
                                                 - audio_start_timestamp) // Constants.BLOCK_SIZE)
        processed_audio_data['student'][processed_audio_frame['frameNumber']] = processed_audio_frame

    t_process_audio_data_end = datetime.now()

    # Loop over each frame and reformat frame data for instructor side camera

    audioFrames = raw_audio_data['instructor']['sessions'][0]['audioFrames']

    for audioFrame in audioFrames:
        processed_audio_frame = preprocess_audio_frame(audioFrame, logger)
        processed_audio_frame['block_id'] = int(((processed_audio_frame['timestamp'] / Constants.MILLISECS_IN_SEC)
                                                 - audio_start_timestamp) // Constants.BLOCK_SIZE)
        processed_audio_data['instructor'][processed_audio_frame['frameNumber']] = processed_audio_frame

    t_process_audio_data_end = datetime.now()

    logger.info("Processed Raw Audio Data in | %.3f secs.",
                time_diff(t_process_audio_data_start, t_process_audio_data_end))

    return processed_audio_data


def audio_quality_metrics(session_input_object, processed_audio_data, logger_pass):
    """
    Reports quality metrics for audio data received from

    Parameters:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        processed_audio_data(dict)     : Dictionary containing audio data processed for analytics engine
        logger_pass(logger)            : The parent logger from which to derive the logger for the engine

    Returns:
        audio_quality_metrics(dict)    : Dictionary containing quality assessment of audio data going
                                         into analytics engine
    """

    return {}


def preprocess_audio_frame(raw_audio_frame, logger):
    """
    process and structure single audio frame

    Parameters:
        raw_audio_frame(dict)          : Raw audio information of one frame of session
        logger(logger)                 : Inherited logger from parent function

    Returns:
        processed_audio_frame(dict)    : Processed audio information of one frame of session
    """

    processed_audio_frame = dict()
    processed_audio_frame['frameNumber'] = raw_audio_frame['frameNumber']

    # get timestamp information

    processed_audio_frame['timestamp'] = int((raw_audio_frame['timestamp']['unixSeconds'] * Constants.NANOSECS_IN_SEC +
                                              raw_audio_frame['timestamp']['unixNanoseconds']) // 1e6)
    processed_audio_frame['asctime'] = raw_audio_frame['timestamp']['RFC3339']

    # get inference information

    processed_audio_frame['amplitude'] = raw_audio_frame['audio']['amplitude']
    processed_audio_frame['melFrequency'] = raw_audio_frame['audio']['melFrequency']

    if 'mfccFeatures' in raw_audio_frame['audio'].keys():
        processed_audio_frame['mfccFeatures'] = raw_audio_frame['audio']['mfccFeatures']
    else:
        processed_audio_frame['mfccFeatures'] = None
        # logger.warning("MFCC features not found for audio. Check input schema version..")

    if 'polyFeatures' in raw_audio_frame['audio'].keys():
        processed_audio_frame['polyFeatures'] = raw_audio_frame['audio']['polyFeatures']
    else:
        processed_audio_frame['polyFeatures'] = None
        # logger.warning("Poly Spectral features not found for audio. Check input schema version..")

    return processed_audio_frame

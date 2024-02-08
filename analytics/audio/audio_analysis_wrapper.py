"""
Author: Anonymized
Created: Anonymized

This file contains wrapper function to run end to end analysis on audio data
"""

# Import python library functions
import logging
from datetime import datetime

# Import project level functions and classes
from configs.constants import exitStatus
from utils.time_utils import time_diff
from analytics.audio.silence_detection import run_silence_detection_module
from analytics.audio.object_noise_detection import run_object_noise_detection_module
from analytics.audio.teacher_speech_detection import run_teacher_speaking_module
from analytics.audio.single_speaker_detection import run_single_speaker_module


def audio_analysis_wrapper(session_input_object, session_output_object, logger_pass):
    """
    This is main wrapper function to run all audio modules

    Parameters:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        session_output_object(dict)    : Dictionary to collect all outputs for session analytics
        logger_pass(logger)            : The parent logger from which to derive the logger for this function

    Returns:
        session_output_object(dict)    : Dictionary to collect all outputs for session analytics
    """
    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('audio_analysis')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_audio_analysis_start = datetime.now()

    # Initialize audio module

    session_output_object['audio'] = {
        'second': dict(),
        'block': dict(),
        'session': dict(),
        'debug': dict()
    }

    # Run silence detection modules at second, block and session level

    silence_detection_info, run_status = run_silence_detection_module(session_input_object, session_output_object,
                                                                      logger_pass)

    if not (run_status == exitStatus.SUCCESS):
        logger.warning("Silence module did not execute successfully")
    else:
        session_output_object['audio']['second'].update(silence_detection_info['second'])
        session_output_object['audio']['block'].update(silence_detection_info['block'])
        session_output_object['audio']['session'].update(silence_detection_info['session'])
        session_output_object['audio']['debug'].update(silence_detection_info['debug'])

    # Run object noise detection module at second, block and session level

    object_noise_detection_info, run_status = run_object_noise_detection_module(session_input_object,
                                                                                session_output_object, logger_pass)

    if not (run_status == exitStatus.SUCCESS):
        logger.warning("Object noise detection module did not execute successfully")
    else:
        session_output_object['audio']['second'].update(object_noise_detection_info['second'])
        session_output_object['audio']['block'].update(object_noise_detection_info['block'])
        session_output_object['audio']['session'].update(object_noise_detection_info['session'])

    # Run teacher speech detection module at second, block and session level

    teacher_speech_detection_info, run_status = run_teacher_speaking_module(session_input_object, session_output_object,
                                                                            logger_pass)

    if not (run_status == exitStatus.SUCCESS):
        logger.warning("Teacher speech detection module did not execute successfully")
    else:
        session_output_object['audio']['second'].update(teacher_speech_detection_info['second'])
        session_output_object['audio']['block'].update(teacher_speech_detection_info['block'])
        session_output_object['audio']['session'].update(teacher_speech_detection_info['session'])

    # Run single speaker detection module at second, block and session level

    single_speaker_detection_info, run_status = run_single_speaker_module(session_input_object, session_output_object,
                                                                          logger_pass)

    if not (run_status == exitStatus.SUCCESS):
        logger.warning("Single Speaker detection module did not execute successfully")
    else:
        session_output_object['audio']['second'].update(single_speaker_detection_info['second'])
        session_output_object['audio']['block'].update(single_speaker_detection_info['block'])
        session_output_object['audio']['session'].update(single_speaker_detection_info['session'])

    t_audio_analysis_end = datetime.now()

    logger.info("Audio Analysis took | %.3f secs.",
                time_diff(t_audio_analysis_start, t_audio_analysis_end))

    return session_output_object

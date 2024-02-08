"""
Author: Anonymized
Created: Anonymized

This file contains functions to fetch raw audio and video data from graphql backend
"""

import logging
from datetime import datetime

# Import project level functions and classes
from utils.time_utils import time_diff
from configs.graphql_query_schemas import build_query
from drivers.edusense.utils.fetch_graphql_data import fetch_graphql_data


def fetch_audio_data(session_input_object, logger_pass):
    """
    Fetches raw audio information for teacher, student channel from graphql backend

    Parameters:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        logger_pass(logger)            : The parent logger from which to derive the logger for the engine

    Returns:
        raw_audio_data(dict)           : Dictionary containing audio data in raw graphql format
    """

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('fetch_audio_data')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_fetch_audio_data_start = datetime.now()

    # Build https request to get instructor audio data

    audio_fetch_query = build_query('audio',
                                    session_id=session_input_object.get('session_id'),
                                    channel='instructor',
                                    logger=logger)
    audio_fetch_request = {'query': audio_fetch_query}

    # post request to get audio data

    instructor_audio_data_raw = fetch_graphql_data(audio_fetch_request,
                                                   session_input_object,
                                                   logger_pass,
                                                   credentials_prefix="")

    # Build https request to get student audio data

    audio_fetch_query = build_query('audio',
                                    session_id=session_input_object.get('session_id'),
                                    channel='student',
                                    logger=logger)
    audio_fetch_request = {'query': audio_fetch_query}

    # post request to get audio data

    student_audio_data_raw = fetch_graphql_data(audio_fetch_request,
                                                session_input_object,
                                                logger_pass,
                                                credentials_prefix="")

    raw_audio_data = {
        'student': student_audio_data_raw,
        'instructor': instructor_audio_data_raw
    }

    t_fetch_audio_data_end = datetime.now()

    logger.info("Fetched Raw Audio Data in | %.3f secs.", time_diff(t_fetch_audio_data_start, t_fetch_audio_data_end))

    return raw_audio_data


def fetch_video_data(session_input_object, logger_pass):
    """
    Fetches raw video information for teacher, student channel from graphql backend

    Parameters:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        logger_pass(logger)            : The parent logger from which to derive the logger for the engine

    Returns:
        raw_video_data(dict)           : Dictionary containing video data in raw graphql format
    """

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('fetch_video_data')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_fetch_video_data_start = datetime.now()

    # Build https request to get instructor video data

    video_fetch_query = build_query('video',
                                    session_id=session_input_object.get('session_id'),
                                    channel='instructor',
                                    logger=logger)
    video_fetch_request = {'query': video_fetch_query}

    # post request to get video data

    instructor_video_data_raw = fetch_graphql_data(video_fetch_request, session_input_object, logger_pass)

    # Build https request to get student video data

    video_fetch_query = build_query('video',
                                    session_id=session_input_object.get('session_id'),
                                    channel='student',
                                    logger=logger)
    video_fetch_request = {'query': video_fetch_query}

    # post request to get video data

    student_video_data_raw = fetch_graphql_data(video_fetch_request, session_input_object, logger_pass)

    raw_video_data = {
        'student': student_video_data_raw,
        'instructor': instructor_video_data_raw
    }

    t_fetch_video_data_end = datetime.now()

    logger.info("Fetched Raw Video Data in | %.3f secs.", time_diff(t_fetch_video_data_start, t_fetch_video_data_end))

    return raw_video_data

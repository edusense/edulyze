"""
Author: Anonymized
Created: Anonymized

This file restructures video input data in format required by analytics engine
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Import project level functions and classes
from configs.constants import Constants
from utils.time_utils import time_diff


def preprocess_video_data(session_input_object, raw_video_data, logger_pass):
    """
    Fetches raw video information for teacher, student channel from graphql backend

    Parameters:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        raw_video_data(dict)           : Dictionary containing video data in raw graphql format
        logger_pass(logger)            : The parent logger from which to derive the logger for the engine

    Returns:
        processed_video_data(dict)      : Dictionary containing video data processed for analytics engine
    """

    # initialize logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('preprocess_video_data')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_preprocess_video_data_start = datetime.now()

    # initialize processed video data

    processed_video_data = dict({
        'student': {},
        'instructor': {}
    })

    # Add None Ids count in session input object to handle non tracking ids

    session_input_object['none_tracking_id_count'] = 0

    # Loop over each frame and reformat frame data for student side camera

    videoFrames = raw_video_data['student']['sessions'][0]['videoFrames']
    student_video_start_timestamp = int(
        raw_video_data['student']['sessions'][0]['videoFrames'][0]['timestamp']['unixSeconds'])

    for videoFrame in videoFrames:
        processed_video_frame = preprocess_video_frame(session_input_object, videoFrame, logger)
        processed_video_frame['block_id'] = int(((processed_video_frame['timestamp'] / Constants.MILLISECS_IN_SEC)
                                                 - student_video_start_timestamp) // Constants.BLOCK_SIZE)
        processed_video_data['student'][processed_video_frame['frameNumber']] = processed_video_frame

    t_process_video_data_end = datetime.now()

    # Loop over each frame and reformat frame data for instructor side camera

    videoFrames = raw_video_data['instructor']['sessions'][0]['videoFrames']
    instructor_video_start_timestamp = int(
        raw_video_data['instructor']['sessions'][0]['videoFrames'][0]['timestamp']['unixSeconds'])
    for videoFrame in videoFrames:
        processed_video_frame = preprocess_video_frame(session_input_object, videoFrame, logger)
        processed_video_frame['block_id'] = int(((processed_video_frame['timestamp'] / Constants.MILLISECS_IN_SEC)
                                                 - instructor_video_start_timestamp) // Constants.BLOCK_SIZE)
        processed_video_data['instructor'][processed_video_frame['frameNumber']] = processed_video_frame

    t_preprocess_video_data_end = datetime.now()

    logger.info("Processed Raw Video Data in | %.3f secs.",
                time_diff(t_preprocess_video_data_start, t_preprocess_video_data_end))

    return processed_video_data


def video_quality_metrics(session_input_object, processed_video_data, logger_pass):
    """
    Reports quality metrics for video data received from

    Parameters:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        processed_video_data(dict)     : Dictionary containing video data processed for analytics engine
        logger_pass(logger)            : The parent logger from which to derive the logger for the engine

    Returns:
        video_quality_metrics(dict)    : Dictionary containing quality assessment of video data going
                                         into analytics engine
    """

    return {}


def preprocess_video_frame(session_input_object, raw_video_frame, logger):
    """
    process and structure single video frame

    Parameters:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        raw_video_frame(dict)          : Raw video information of one frame of session
        logger(logger)                 : Inherited logger from parent function

    Returns:
        processed_video_frame(dict)    : Processed video information of one frame of session
    """

    processed_video_frame = dict()
    processed_video_frame['frameNumber'] = raw_video_frame['frameNumber']

    # get timestamp information

    processed_video_frame['timestamp'] = int((raw_video_frame['timestamp']['unixSeconds'] * Constants.NANOSECS_IN_SEC +
                                              raw_video_frame['timestamp']['unixNanoseconds']) // 1e6)
    processed_video_frame['asctime'] = raw_video_frame['timestamp']['RFC3339']

    # get people information

    peopleFrameData = raw_video_frame['people']
    processed_video_frame['people'] = dict()
    min_tracking_id = -1
    for personData in peopleFrameData:
        processed_person_data = preprocess_person_data(personData, logger)
        if processed_person_data['trackingId'] is None:
            # Todo: Complete this part gracefully. It is just a workaround to skip students
            session_input_object['none_tracking_id_count'] += 1
            processed_person_data['trackingId'] = min_tracking_id
            min_tracking_id -=1
            # processed_person_data['trackingId'] = (-1) * session_input_object['none_tracking_id_count']
            processed_video_frame['people'][processed_person_data['trackingId']] = processed_person_data
        else:
            processed_video_frame['people'][processed_person_data['trackingId']] = processed_person_data

    return processed_video_frame


def preprocess_person_data(raw_person_data, logger):
    """
    process and structure single person data from a video frame

    Parameters:
        raw_person_data(dict)          : Raw information of one person from one of the frames of session
        logger(logger)                 : Inherited logger from parent function

    Returns:
        processed_person_data(dict)    : Processed information of one person from one of the frame of session
    """

    processed_person_data = dict()

    # Get Keypoints information

    processed_person_data['body_kps'] = raw_person_data['body']
    processed_person_data['face_kps'] = raw_person_data['face']
    processed_person_data['hand_kps'] = raw_person_data['hand']

    # Get Id Information

    processed_person_data['trackingId'] = raw_person_data['inference']['trackingId']

    # Get posture, head, and face info

    processed_person_data.update(raw_person_data['inference']['posture'])
    processed_person_data.update(raw_person_data['inference']['face'])
    processed_person_data.update(raw_person_data['inference']['head'])

    return processed_person_data

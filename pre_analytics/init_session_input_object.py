"""
Author: Anonymized
Created: Anonymized

This file contains wrapper function to prepare input for analytics pipeline
"""

import logging
import os

import pandas as pd
from datetime import datetime
import pickle

# Import project level functions and classes
from configs.constants import exitStatus
from pii_information.location_config import get_location_config
from utils.time_utils import time_diff, extract_timestamp
from drivers.edusense.utils.fetch_graphql_data import NoInstructorDataError, NoStudentDataError
from drivers.DriverInterface import MethodNotImplementedException
from preprocessing.location import preprocess_location_data
from preprocessing.gaze import preprocess_gaze_data
from preprocessing.audio import preprocess_audio_data



# from pre_analytics.fetch_input import fetch_audio_data
# from pre_analytics.fetch_input import fetch_video_data
# from pre_analytics.preprocess_audio_data import preprocess_audio_data, audio_quality_metrics
# from pre_analytics.preprocess_video_data import preprocess_video_data, video_quality_metrics
# from pre_analytics.impute_audio_data import impute_audio_data
# from pre_analytics.impute_video_data import impute_video_data, location_based_sync_student_ids


def init_session_input_object(run_config, input_driver, logger_pass):
    """
    Wrapper function to create input object for session analytics based on given input driver

    Parameters:
        run_config(dict)               : Dictionary containing all attributes needed to run session analysis
        input_driver(DriverInterface)  : Input driver for underlying sensing system
        logger_pass(logger)            : The parent logger from which to derive the logger for the engine

    Returns:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
    """

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('init_session_input_object')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_init_session_input_object_start = datetime.now()

    # initialize session input object

    session_input_object = dict()
    session_input_object.update(run_config)

    # Get meta data for session
    try:
        raw_meta_data_dict = input_driver.getMetaInput()
    except MethodNotImplementedException:
        logger.warning("Meta input not provided by driver")
        raw_meta_data_dict = dict()


    # get location information for session
    try:
        df_raw_location_data = input_driver.getLocationInput()
        df_processed_location_data, location_preprocessing_metrics = preprocess_location_data(df_raw_location_data, raw_meta_data_dict, logger)
    except MethodNotImplementedException:
        logger.warning("Location input not provided by driver")
        df_processed_location_data, location_preprocessing_metrics = None, None


    # get gaze information for session
    try:
        df_raw_gaze_data = input_driver.getGazeInput()
        df_processed_gaze_data, gaze_preprocessing_metrics = preprocess_gaze_data(df_raw_gaze_data, raw_meta_data_dict, logger)
    except MethodNotImplementedException:
        logger.warning("Gaze input not provided by driver")
        df_raw_gaze_data = None

    # Get audio data for session
    try:
        df_raw_audio_data = input_driver.getAudioInput()
        df_processed_audio_data, audio_preprocessing_metrics = preprocess_audio_data(df_raw_audio_data, raw_meta_data_dict, logger)
    except MethodNotImplementedException:
        logger.warning("Audio input not provided by driver")
        df_raw_audio_data = None

    # Fetch video data for session

    # raw_video_data = fetch_video_data(session_input_object, logger_pass)
    #
    # # Get start timestamp from session keyword for block assignment
    #
    # session_keyword = session_input_object.get('session_keyword')
    # session_start_timestamp = extract_timestamp(session_keyword, logger)
    #
    # if session_start_timestamp == -1:
    #     # fetch time from first audio frame
    #     logger.warning("Unable to extract start timestamp from session keyword, using first student videoframe time")
    #     session_start_timestamp = int(
    #         raw_video_data['student']['sessions'][0]['videoFrames'][0]['timestamp']['unixSeconds'])
    #
    # session_input_object['session_start_timestamp'] = session_start_timestamp
    #
    # # Process and restructure audio data
    #
    # processed_audio_data = preprocess_audio_data(session_input_object, raw_audio_data, logger_pass)
    # session_input_object['audio'] = dict()
    # session_input_object['audio']['raw_data_quality_metrics'] = audio_quality_metrics(session_input_object,
    #                                                                                   processed_audio_data, logger_pass)
    #
    # # Process and restructure video data
    #
    # processed_video_data = preprocess_video_data(session_input_object, raw_video_data, logger_pass)
    # session_input_object['video'] = dict()
    # session_input_object['video']['raw_data_quality_metrics'] = video_quality_metrics(session_input_object,
    #                                                                                   processed_video_data, logger_pass)
    #
    # # fill missing audio data
    #
    # imputed_audio_data, imputation_metrics = impute_audio_data(session_input_object, processed_audio_data, logger_pass)
    # session_input_object['audio'].update(imputed_audio_data)
    # session_input_object['audio']['imputation_metrics'] = imputation_metrics
    #
    # # fill missing video data and sync tracking IDs for student frames
    #
    # imputed_video_data, imputation_metrics = impute_video_data(session_input_object, processed_video_data, logger_pass)
    # session_input_object['video'].update(imputed_video_data)
    # session_input_object['video']['imputation_metrics'] = imputation_metrics
    #
    # # Todo: Removed for now because we do not need audio df anywhere directly
    # # Create dataframe input out of session audio input
    #
    # # df_student_audio_input = pd.DataFrame.from_dict(session_input_object['audio']['student'], orient='index')
    # # df_student_audio_input['channel'] = 'student'
    # # df_instructor_audio_input = pd.DataFrame.from_dict(session_input_object['audio']['instructor'], orient='index')
    # # df_instructor_audio_input['channel'] = 'instructor'
    # # df_audio_input = pd.concat([df_student_audio_input, df_instructor_audio_input]).sort_values(
    # #     by=['timestamp', 'channel']).reset_index(drop=True)
    # #
    # # session_input_object['input_audio_df'] = df_audio_input
    #
    # # Create dataframe input out of session video input
    #
    # df_student_video_input = create_video_dataframe(session_input_object['video']['student'], logger)
    # df_student_video_input['channel'] = 'student'
    #
    # # sync student ids based on frame level switches and location proxemics
    # df_student_video_input = location_based_sync_student_ids(df_student_video_input, logger)
    # df_instructor_video_input = create_video_dataframe(session_input_object['video']['instructor'], logger)
    # df_instructor_video_input['channel'] = 'instructor'
    #
    # # Check if there exists any body for instructor or student
    #
    # if df_student_video_input.shape[0] == 0:
    #     logger.error('No student present in class for whole recorded session, Exiting analytics pipeline..')
    #     raise NoStudentDataError
    #
    # if df_instructor_video_input.shape[0] == 0:
    #     logger.error('Instructor not present in class for whole recorded session, Exiting analytics pipeline..')
    #     raise NoInstructorDataError
    #
    # df_video_input = pd.concat([df_student_video_input, df_instructor_video_input]).sort_values(
    #     by=['timestamp', 'channel', 'trackingId']).reset_index(drop=True)
    #
    # session_input_object['input_video_df'] = df_video_input
    #
    # # get location config for given classroom
    #
    # location_config_file = session_input_object.get("location_config_file", None)
    # classroom = "_".join(session_keyword.split("_")[-3:-1])
    #
    # location_config = get_location_config(classroom, logger, location_config_file)
    # session_input_object['location_config'] = location_config

    # df_location_gaze_data = pd.merge(df_processed_gaze_data,df_processed_location_data,on=['frameNumber',
    #                                                                                        'timestamp',
    #                                                                                        'channel',
    #                                                                                        'block_id',
    #                                                                                        'trackingI'])
    session_input_object.update({
        'input_audio_df':df_raw_audio_data,
        'input_location_df':df_processed_location_data,
        'input_gaze_df':df_raw_gaze_data,
        'session_meta_data':raw_meta_data_dict
    })

    # # Cache Data if cache mode is true
    #
    # if session_input_object.get("cache_mode"):
    #     cache_status = cache_session_data(session_input_object, logger)
    #     if cache_status == exitStatus.SUCCESS:
    #         logger.info("Cached Session Data Successfully")
    #     else:
    #         logger.error("Caching session data failed with status | %s", cache_status.name)

    t_init_session_input_object_end = datetime.now()

    logger.info("Initialization of input object took | %.3f secs.",
                time_diff(t_init_session_input_object_start, t_init_session_input_object_end))

    return session_input_object


def create_video_dataframe(video_data, logger):
    """
    Create dataframe out of dictionary format input

    Parameters:
        video_data(dict)               : Dictionary containing processed and imputed video data
        logger_pass(logger)            : The parent logger from which to derive the logger for the engine

    Returns:
        video_df(pd.DataFrame)         : Dataframe format for given dictionary video data
    """

    t_create_video_df_start = datetime.now()

    # loop over all frames and compile video dfs
    frame_df_list = []

    for frameNumber, frame_video_dict in video_data.items():

        people_ids = list(frame_video_dict['people'].keys())

        if len(people_ids) == 0:  # noone detected in this frame, go to next frame
            # logger.debug(f"Noone detected in frameNumber: {frameNumber}. skipping for df input.")
            continue

        # create dict out of people information
        frame_df = pd.DataFrame.from_dict(frame_video_dict['people'], orient='index')

        # add columns for frame information
        frame_df['frameNumber'] = frameNumber
        frame_df['timestamp'] = frame_video_dict['timestamp']
        frame_df['asctime'] = frame_video_dict['asctime']
        frame_df['block_id'] = frame_video_dict['block_id']

        # add frame_df to main_df
        frame_df_list.append(frame_df)

    # compile all frame dfs into one single dataframe
    if len(frame_df_list) > 0:
        video_df = pd.concat(frame_df_list)
    else:
        video_df = pd.DataFrame()
        logger.error("No Frame level dataframes present, returning empty dataFrame")

    t_create_video_df_end = datetime.now()

    logger.info("Creating video dataframe took | %.3f secs. ",
                time_diff(t_create_video_df_start, t_create_video_df_end))

    return video_df


def cache_session_data(session_input_object, logger):
    """
    Cache complete session input object for each seesion in pickle format

    Parameters:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        logger(logger)                 : The parent logger to log in function

    Returns:
        cache_status(exitStatus)       : Exit flag for caching success
    """
    t_cache_data_start = datetime.now()

    cache_dir = session_input_object.get('cache_dir')
    server_name = session_input_object.get('server_name')
    session_id = session_input_object.get('session_id')

    cache_status = exitStatus.SUCCESS
    try:
        if not os.path.exists(f"{cache_dir}/{server_name}"):
            os.makedirs(f"{cache_dir}/{server_name}")
        with open(f"{cache_dir}/{server_name}/{session_id}.pb", "wb") as f:
            pickle.dump(session_input_object, f)
    except:

        cache_status = exitStatus.FAILURE

    t_cache_data_end = datetime.now()

    logger.info("Caching Session Input took | %.3f secs.", time_diff(t_cache_data_start, t_cache_data_end))

    return cache_status

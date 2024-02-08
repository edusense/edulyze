"""
Author: Anonymized
Created: Anonymized

Main run file to run session analytics for Edulyze
"""

# Import python library functions
import sys
import os
import logging
from datetime import datetime
from logging.handlers import WatchedFileHandler
import traceback
import pandas as pd
import csv
import jstyleson as json
import argparse

# Import project level functions and classes
from configs.constants import Constants, exitStatus
from utils.time_utils import time_diff
from drivers.edusense.utils.fetch_graphql_data import NoInstructorDataError, NoStudentDataError
from pre_analytics.init_session_input_object import init_session_input_object
from pre_analytics.init_session_output_object import init_session_output_object
from analytics.audio.audio_analysis_wrapper import audio_analysis_wrapper
from analytics.location.location_analysis_wrapper import location_analysis_wrapper
from analytics.gaze.gaze_analysis_wrapper import gaze_analysis_wrapper
from post_analytics.create_output_payload import create_output_payload
from post_analytics.post_results_to_backend import post_results_to_backend
from post_analytics.post_results_to_file import post_results_to_file, output_exists_in_cache
from drivers import get_driver


def run_single_session_analysis(run_config, root_logger):
    """
    Main function to session analysis for single session

    Parameters:
        run_config(dict)               : Dictionary containing all attributes needed to run session analysis
        root_logger(logger)            : The parent logger from which to derive the logger for this function

    Returns:
        run_status(int)                : Status of whether analysis run is complete for session
    """

    # Initialize the logger

    logger_base = root_logger.getChild('run_single_session')
    logging_dict = {
        "session_id": run_config.get('session_id', run_config.get('video_session_id', None)),
        "session_keyword": run_config.get("session_keyword")
    }
    logger = logging.LoggerAdapter(logger_base, logging_dict)

    logger_pass = {
        'logger_base': logger_base,
        'logging_dict': logging_dict
    }

    t_run_single_session_analysis_start = datetime.now()
    run_config.update({'analysis_start_time': int(t_run_single_session_analysis_start.strftime("%s"))})

    logger.info("------------------------------------------")
    logger.info("Starting single session analysis.")

    # Test if output exists in cache
    if output_exists_in_cache(run_config, logger_pass):
        # Exits pipeline gracefully
        logger.info("Required output exists in cache. Not running pipeline.")
        post_results_status = exitStatus.RESULTS_CACHED
        logger.info("Single session analytics execution completed with status | %s",
                    post_results_status.name)
        t_run_single_session_analysis_end = datetime.now()

        logger.info("Running session analysis took | %.3f secs.",
                    time_diff(t_run_single_session_analysis_start, t_run_single_session_analysis_end))
        logger.info("------------------------------------------")

        return post_results_status

    # initialize driver for input

    input_driver_class = get_driver(run_config.get('driver'))

    logger_driver_base = logger_pass.get('logger_base').getChild(f'init_{run_config.get("driver")}_driver')
    logger_driver = logging.LoggerAdapter(logger_driver_base, logger_pass.get('logging_dict'))
    input_driver = input_driver_class(run_config, logger_driver)

    # Create session_input_object

    session_input_object = init_session_input_object(run_config, input_driver, logger_pass)

    # Create session_output_object

    session_output_object = init_session_output_object(run_config, logger_pass)

    # Run Single Modal Analysis

    session_output_object = run_single_modality_analysis(session_input_object, session_output_object, logger_pass)

    # Validate and Post Results
    session_output_object.update(
        {'analysis_run_time': int(time_diff(t_run_single_session_analysis_start, datetime.now()))})

    post_results_status = post_session_results(session_input_object, session_output_object, logger_pass)

    if post_results_status == exitStatus.SUCCESS:
        logger.info("Single session analytics execution completed with status | %s",
                    post_results_status.name)
    else:
        logger.error("Single session analytics failed with status | %s",
                     post_results_status.name)

    t_run_single_session_analysis_end = datetime.now()

    logger.info("Running session analysis took | %.3f secs.",
                time_diff(t_run_single_session_analysis_start, t_run_single_session_analysis_end))
    logger.info("------------------------------------------")

    return post_results_status


def run_single_modality_analysis(session_input_object, session_output_object, logger_pass):
    """
    Run single modality analysis for one session. it includes gaze, location, audio and posture

    Parameters:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        session_output_object(dict)    : Dictionary to collect all outputs for session analytics
        logger_pass(logger)            : The parent logger from which to derive the logger for this function

    Returns:
        session_output_object(dict)    : Dictionary to collect all outputs for session analytics
    """

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('single_modality_analysis')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_single_modality_analysis_start = datetime.now()

    # TODO: Add wrapper functions here for cross modality analysis

    # Run Audio Analysis
    t_audio_module_start = datetime.now()
    try:
        session_output_object['run_modules'].append('audio')
        session_output_object = audio_analysis_wrapper(session_input_object, session_output_object, logger_pass)
        t_audio_module_end = datetime.now()
        session_output_object['module_runtimes'].append(int(time_diff(t_audio_module_start, t_audio_module_end)))
        session_output_object['success_modules'].append('audio')
    except:
        logger.error("Error in running audio pipeline | %s", traceback.format_exc())
        t_audio_module_end = datetime.now()
        session_output_object['module_runtimes'].append(int(time_diff(t_audio_module_start, t_audio_module_end)))
        session_output_object['failure_modules'].append('audio')

    # Run Gaze Analysis
    t_gaze_module_start = datetime.now()
    try:
        session_output_object['run_modules'].append('gaze')
        session_output_object = gaze_analysis_wrapper(session_input_object, session_output_object, logger_pass)
        t_gaze_module_end = datetime.now()
        session_output_object['module_runtimes'].append(int(time_diff(t_gaze_module_start, t_gaze_module_end)))
        session_output_object['success_modules'].append('gaze')
    except:
        logger.error("Error in running gaze pipeline | %s", traceback.format_exc())
        t_gaze_module_end = datetime.now()
        session_output_object['module_runtimes'].append(int(time_diff(t_gaze_module_start, t_gaze_module_end)))
        session_output_object['failure_modules'].append('gaze')

    # Run Location Analysis
    t_location_module_start = datetime.now()
    try:
        session_output_object['run_modules'].append('location')
        session_output_object = location_analysis_wrapper(session_input_object, session_output_object, logger_pass)
        t_location_module_end = datetime.now()
        session_output_object['module_runtimes'].append(int(time_diff(t_location_module_start, t_location_module_end)))
        session_output_object['success_modules'].append('location')
    except:
        logger.error("Error in running location pipeline | %s", traceback.format_exc())
        t_location_module_end = datetime.now()
        session_output_object['module_runtimes'].append(int(time_diff(t_location_module_start, t_location_module_end)))
        session_output_object['failure_modules'].append('location')

    t_single_modality_analysis_end = datetime.now()

    logger.info("Single Modality Analysis took | %.3f secs.",
                time_diff(t_single_modality_analysis_start, t_single_modality_analysis_end))

    return session_output_object


def post_session_results(session_input_object, session_output_object, logger_pass):
    """
    Post results for single session analysis to json file, or storage backend based on configurations

    Parameters:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        session_output_object(dict)    : Dictionary to collect all outputs for session analytics
        logger_pass(logger)            : The parent logger from which to derive the logger for this function

    Returns:
        post_results_status(exitStatus): Exit status code for posting analysis results
    """
    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('post_session_results')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_post_session_results_start = datetime.now()
    post_results_status = exitStatus.SUCCESS
    success_post_status_count = 0

    # Create output payload for posting on analytics storage backend

    output_payload = create_output_payload(session_input_object, session_output_object, logger_pass)

    # post results to analytics backend

    post_results_backend_success = post_results_to_backend(session_input_object, session_output_object,
                                                           output_payload, logger_pass)
    post_results_backend_success = exitStatus.FAILURE

    if post_results_backend_success == exitStatus.SUCCESS:
        logger.info("Session analytics posting to backend completed with status | %s",
                    post_results_backend_success.name)
        success_post_status_count += 1
    else:
        logger.error("Session analytics posting to backend not completed with status | %s",
                     post_results_backend_success.name)

    # post df results to files
    dummy_payload = {}
    post_results_file_success = post_results_to_file(session_input_object, session_output_object,
                                                     dummy_payload, logger_pass)

    if post_results_file_success == exitStatus.SUCCESS:
        logger.info("Session analytics posting to file completed with status | %s",
                    post_results_file_success.name)
        success_post_status_count += 1
    else:
        logger.error("Session analytics posting to file failed with status | %s",
                     post_results_file_success.name)

    if success_post_status_count == 1:
        post_results_status = exitStatus.PARTIAL_SUCCESS
    elif success_post_status_count == 0:
        post_results_status = exitStatus.FAILURE

    t_post_session_results_end = datetime.now()

    logger.info("Posting single session results took | %.3f secs.",
                time_diff(t_post_session_results_start, t_post_session_results_end))
    return post_results_status


if __name__ == '__main__':

    # Config to run engine
    default_config_file = "examples_configs/edusense_classgaze_single.json"

    parser = argparse.ArgumentParser(description='Run Edulyze analytics pipeline.')
    parser.add_argument('--run_config', type=str, default=default_config_file, help='Path to run config file')
    args = parser.parse_args()
    run_config_file = args.run_config

    run_config = json.load(open(run_config_file))

    # Initialize the logger
    logger_master = logging.getLogger('analytics_engine')
    logger_master.setLevel(logging.DEBUG)

    ## Add core logger handler
    core_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(session_keyword)s | %(session_id)s | %(message)s')
    core_logging_handler = WatchedFileHandler(Constants.LOG_DIR + '/' + Constants.LOG_FILE)
    core_logging_handler.setFormatter(core_formatter)
    logger_master.addHandler(core_logging_handler)

    ## Add stdout logger handler
    console_formatter = logging.Formatter(
        '%(asctime)s | %(module)s | %(levelname)s | %(session_keyword)s | %(session_id)s | %(message)s')
    console_log = logging.StreamHandler()
    console_log.setLevel(logging.DEBUG)
    console_log.setFormatter(console_formatter)
    logger_master.addHandler(console_log)

    if run_config["input_data_type"] == "file":
        # Run analytics pipeline for single session

        # Get username and password from environment variables
        for key in run_config:
            if 'server' in key:
                run_config[key] = os.getenv(run_config[key], run_config[key])
        df_session_data = pd.read_csv(f'{run_config["session_data_file"]}')
        df_session_data['timestamp'] = pd.to_datetime(df_session_data['timestamp'], format='%d/%m/%Y %H:%M:%S')

        df_session_data = df_session_data[df_session_data.phase.astype(int) == 2].groupby('session', as_index=False)[
            'timestamp'].min()
        df_session_data['timestamp'] = df_session_data['timestamp'].dt.strftime("%Y%m%d%H%M%S")

        for session_record in df_session_data.to_dict(orient='records'):
            run_config['session_id'] = session_record['session']
            run_config[
                'session_keyword'] = f"course1_lab1_{session_record['session'].replace(' ', '_').replace('.', '')}_{session_record['timestamp']}"
            # Call single session run
            try:
                run_single_session_analysis(run_config, logger_master)
            except:
                print(f"UNABLE TO RUN SESSION {run_config['session_id']}")
        # run_single_session_analysis(run_config, logger_master)
    elif run_config["run_mode"] == "single":
        # Get username and password from environment variables
        for key in run_config:
            if 'server' in key:
                run_config[key] = os.getenv(run_config[key], run_config[key])
        # Call single session run
        run_single_session_analysis(run_config, logger_master)

    elif run_config["run_mode"] == "single":
        # Run analytics pipeline for multiple sessions together

        # Remove single core logging handler based
        logger_master.removeHandler(core_logging_handler)

        # Setup pipeline run records writer
        # get input configs
        config_filepath = run_config["session_list_filepath"]
        config_dicts = pd.read_csv(config_filepath).to_dict(orient='index')

        # update run config with output variables for initiating records writer
        run_config.update({
            '_exit_status': 'NA',
            '_exit_time': '',
        })
        run_config_header = sorted(list(run_config.keys()))
        run_config_header = [xr for xr in run_config_header if 'password' not in xr]
        records_filename = f"analytics_records/{config_filepath.split('/')[-1].split('.')[0]}_run_records.csv"
        records_file_exists = os.path.exists(records_filename)
        pipeline_run_records_writer = csv.DictWriter(open(records_filename, 'a+'),
                                                     run_config_header, extrasaction='ignore')

        # writing header everytime as a delimiter to differentiate multiple runs
        if not records_file_exists:
            pipeline_run_records_writer.writeheader()
        else:
            pipeline_run_records_writer.writerow({
                xr: "--" for xr in run_config_header
            })

        # Get username and password from environment variables
        for key in run_config:
            if 'server' in key:
                run_config[key] = os.getenv(run_config[key], run_config[key])

        for key in config_dicts:
            session_run_config = dict(run_config)
            session_run_config.update(config_dicts[key])
            # Run session and collect status
            session_run_status = exitStatus.FAILURE

            # Add new Logging handle for this session
            log_filename = f"{session_run_config['session_keyword']}_{session_run_config['session_id']}.log"
            config_logging_handler = WatchedFileHandler(session_run_config['log_dir'] + '/' + log_filename)
            config_logging_handler.setFormatter(core_formatter)
            logger_master.addHandler(config_logging_handler)
            try:
                session_run_status = run_single_session_analysis(session_run_config, logger_master)
            except NoInstructorDataError:
                session_run_status = exitStatus.NO_INSTRUCTOR_DATA
                print(f"No instructor data in {str(config_dicts[key])}, {traceback.format_exc()}")
            except NoStudentDataError:
                session_run_status = exitStatus.NO_STUDENT_DATA
                print(f"No student data in {str(config_dicts[key])}, {traceback.format_exc()}")
            except:
                session_run_status = exitStatus.FAILURE
                print(f"Error in running session {str(config_dicts[key])}, {traceback.format_exc()}")

            session_run_config.update({
                '_exit_status': session_run_status.name,
                '_exit_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            logger_master.removeHandler(config_logging_handler)
            pipeline_run_records_writer.writerow(session_run_config)
    else:
        raise NotImplementedError(
            f"Run mode {run_config['run_mode']} not implemented. only allowed values are single/multiple")

"""
Author: Anonymized
Created: Anonymized
Post analytics fora session in file output mode
"""
# Import python library functions
import sys
import os
import logging
import traceback
from datetime import datetime
import pickle
import json
from itertools import chain

# Import external library functions
import pandas as pd
import statistics
import math
import numpy as np

# Import project level functions and classes
from configs.constants import Constants, exitStatus
from utils.time_utils import time_diff


def post_results_to_file(session_input_object, session_output_object, output_payload, logger_pass):
    """
    This functions posts results to file in required format

    Parameters:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        session_output_object(dict)    : Dictionary to collect all outputs for session analytics
        logger_pass(logger)            : The parent logger from which to derive the logger for this function
        output_payload(json)           : dict payload for final schema
    Returns:
        post_results_status(exitStatus): Exit status code for posting analysis results to file
    """

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('post_results_to_file')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    posting_status = exitStatus.SUCCESS

    t_post_results_file_start = datetime.now()

    # Check output prefix and output dir and make sure path exists
    output_name = session_input_object.get('output_server_name')
    output_dir = session_input_object.get('output_dir', Constants.DEFAULT_OUTPUT_DIR)
    output_prefix = session_input_object.get('output_prefix', 'session_output')
    output_filedir = f"{output_dir}/{output_name}"

    try:
        if not os.path.exists(output_filedir):
            os.makedirs(output_filedir)

        # get filenames for dumping analytics information

        time_str = datetime.now().strftime("%Y%m")
        # time_str = '202107'
        session_id = session_input_object.get('session_id', session_input_object.get('session_id', ''))
        output_filename_prefix = f"{output_prefix}_{session_input_object.get('session_keyword')}_{session_id}_{time_str}"

        # Post Analytics input and output objects into pickle
        if session_input_object.get('debug_mode', False):
            with open(f"{output_filedir}/{output_filename_prefix}_input.pb", "wb") as dump_input_file:
                pickle.dump(session_input_object, dump_input_file)
        else:
            logger.info("Debug mode not True, Skipping posting input to file results")

        with open(f"{output_filedir}/{output_filename_prefix}_output.pb", "wb") as dump_output_file:
            pickle.dump(session_output_object, dump_output_file)

        with open(f"{output_filedir}/{output_filename_prefix}_output_payload.json", "w") as dump_payload_file:
            json.dump(output_payload, dump_payload_file)

    except:
        logger.error(f"Posting results to file failed | {traceback.format_exc()}")
        posting_status = exitStatus.FAILURE

    t_post_results_file_end = datetime.now()

    logger.info("Posting results to file took | %.3f secs.",
                time_diff(t_post_results_file_start, t_post_results_file_end))

    return posting_status


def output_exists_in_cache(run_config, logger_pass):
    """
    This functions checks if results already exists for current run config in file cache

    Parameters:
        run_config(dict)     : Dictionary containing run config for this pipeline invocation
        logger_pass(logger)  : The parent logger from which to derive the logger for this function
    Returns:
        results_exits(bool)  : Returns true if data for given run config is already cached
    """
    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('output_exists_in_cache')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    # Check output prefix and output dir and make sure path exists
    output_name = run_config.get('output_server_name')
    output_dir = run_config.get('output_dir', Constants.DEFAULT_OUTPUT_DIR)
    output_prefix = run_config.get('output_prefix', 'session_output')
    output_filedir = f"{output_dir}/{output_name}"

    if not os.path.exists(output_filedir):
        logger.info("Output directory does not exist. Results do not exist in cache.")
        return False

    logger.info("Output directory exists for cache.")
    time_str = datetime.now().strftime("%Y%m")
    # time_str = '202107'
    session_id = run_config.get('session_id', run_config.get('session_id', ''))
    output_filename_prefix = f"{output_prefix}_{run_config.get('session_keyword')}_{session_id}_{time_str}"

    if not os.path.exists(f"{output_filedir}/{output_filename_prefix}_output.pb"):
        logger.info("Output results do not exist in cache.")
        return False
    logger.info("Output results exists in cache")

    # if not os.path.exists(f"{output_filedir}/{output_filename_prefix}_output_payload.json"):
    #     logger.info("json payload do not exist in cache.")
    #     return False
    # logger.info("Json payload exists in cache")

    if run_config.get('debug_mode', False):
        if not os.path.exists(f"{output_filedir}/{output_filename_prefix}_input.pb"):
            logger.info("Debug mode is true, and Input results do not exist in cache.")
            return False
        else:
            logger.info("Debug mode is true, Input results exist in cache.")

    return True
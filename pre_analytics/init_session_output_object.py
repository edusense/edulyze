"""
Author: Anonymized
Created: Anonymized

This file contains wrapper function to prepare output object for analytics pipeline
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Import project level functions and classes
from configs.constants import Constants
from utils.time_utils import time_diff


def init_session_output_object(run_config, logger_pass):
    """
    Wrapper function to create output object for session analytics

    Parameters:
        run_config(dict)               : Dictionary containing all attributes needed to run session analysis
        logger_pass(logger)            : The parent logger from which to derive the logger for the engine

    Returns:
        session_output_object(dict)     : Dictionary containing placeholders for all outputs for session analytics
    """

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('init_session_output_object')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_init_session_output_object_start = datetime.now()

    # initialize session output object

    session_output_object = dict()

    # add run modules information
    session_output_object.update({
        'run_modules': [],
        'module_runtimes': [],
        'success_modules': [],
        'failure_modules': []
    })

    # add run config keys to session output object(except password objects)

    session_output_object.update(
        {key: run_config[key] for key in run_config.keys() if 'password' not in key}
    )

    # Todo: insert meta analysis information required by all modules

    t_init_session_output_object_end = datetime.now()

    logger.info("Initialization of output object took | %.3f secs.",
                time_diff(t_init_session_output_object_start, t_init_session_output_object_end))

    return session_output_object

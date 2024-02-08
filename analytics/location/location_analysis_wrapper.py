"""
Author: Anonymized
Created: Anonymized

This file contains wrapper function to run end to end analysis on location data
"""

# Import python library functions
import logging
from datetime import datetime

# Import external library functions

# Import project level functions and classes
from utils.time_utils import time_diff
from analytics.location.instructor_location_module import instructor_location_module
from analytics.location.student_location_module import student_location_module


def location_analysis_wrapper(session_input_object, session_output_object, logger_pass):
    """
    This is main wrapper function to run all location modules

    Parameters:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        session_output_object(dict)    : Dictionary to collect all outputs for session analytics
        logger_pass(logger)            : The parent logger from which to derive the logger for this function

    Returns:
        session_output_object(dict)    : Dictionary to collect all outputs for session analytics
    """
    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('location_analysis')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_location_analysis_start = datetime.now()

    # Initialize location in session output object

    session_output_object['location'] = {
        'instructor': dict(),
        'student': dict()
    }
    # Get location config
    location_config = session_input_object['session_meta_data'].get("location_config")
    if location_config is None:
        logger.error("Location config not available, not running any instructor location module")
        return session_output_object

    # Run instructor location modules

    instructor_location_results = instructor_location_module(session_input_object, session_output_object, logger_pass)
    session_output_object['location']['instructor'].update(instructor_location_results)

    # Run student location modules
    student_location_results = student_location_module(session_input_object, session_output_object, logger_pass)
    session_output_object['location']['student'].update(student_location_results)

    t_location_analysis_end = datetime.now()

    logger.info("Overall Location Analysis took | %.3f secs.",
                time_diff(t_location_analysis_start, t_location_analysis_end))

    return session_output_object

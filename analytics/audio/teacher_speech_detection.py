"""
Author: Anonymized
Created: Anonymized

This file contains code to detect teacher only speaking times in classes at second level, and create statistics at block level.
"""

# Import python library functions
import sys
import os
import logging
from datetime import datetime

# Import project level functions and classes
from configs.constants import Constants, exitStatus
from utils.time_utils import time_diff


def run_teacher_speaking_module(session_input_object, session_output_object, logger_pass):
    """
    This is main function to run teacher speaking detection audio modules

    Parameters:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        session_output_object(dict)    : Dictionary to collect all outputs for session analytics
        logger_pass(logger)            : The parent logger from which to derive the logger for this function

    Returns:
        session_output_object(dict)    : Dictionary to collect all outputs for session analytics
    """
    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('teacher_speaking_module')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_teacher_speaking_start = datetime.now()

    # Initialize silence detection module

    teacher_speaking_results = {
        'second': dict(),
        'block': dict(),
        'session': dict()
    }

    run_status = exitStatus.SUCCESS

    # Todo: Add main code here

    t_teacher_speaking_end = datetime.now()

    logger.info("Teacher speaking detection module took | %.3f secs.",
                time_diff(t_teacher_speaking_start, t_teacher_speaking_end))

    return teacher_speaking_results, run_status

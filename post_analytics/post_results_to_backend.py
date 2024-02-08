"""
Author: Anonymized
Created: Anonymized
Post analytics for a session into mongo storage in backend output mode
"""
# Import python library functions
import sys
import os
import json
import logging
import traceback
from datetime import datetime
from itertools import chain
import requests

# Import external library functions
import pandas as pd
import statistics
import math
import numpy as np
import base64

# Import project level functions and classes
from configs.constants import Constants, exitStatus
from utils.time_utils import time_diff


def post_results_to_backend(session_input_object, session_output_object, output_payload, logger_pass):
    """
    This functions posts results to backend mongo storage in required format

    Parameters:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        session_output_object(dict)    : Dictionary to collect all outputs for session analytics
        logger_pass(logger)            : The parent logger from which to derive the logger for this function
        output_payload(dict)           : Final output payload
    Returns:
        post_results_status(exitStatus): Exit status code for posting analysis results to backend
    """

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('post_results_to_backend')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    posting_status = exitStatus.SUCCESS

    t_post_results_backend_start = datetime.now()

    # ---posting results---

    # credentials for backend posting

    output_server_user = session_input_object.get('output_server_user')
    output_server_password = session_input_object.get('output_server_password')
    output_server_url = f"https://{session_input_object.get('output_server_url')}/analytics"

    cred = '{}:{}'.format(output_server_user, output_server_password).encode('ascii')
    encoded_cred = base64.standard_b64encode(cred).decode('ascii')

    # Query template and posting results

    backend_params = {
        'headers': {
            'Authorization': 'Basic {}'.format(encoded_cred),
            'Content-Type': 'application/json'}
    }

    try:
        posting_response = requests.post(output_server_url, headers=backend_params['headers'], json={'analytics': output_payload})
    except:
        logger.error(f"Posting results to backend failed with error | {traceback.format_exc()}")
        posting_status = exitStatus.FAILURE

    # ---verifying result posting by checking response code---


    # Testing if posting was successfull by extracting meta info and comparing


    t_post_results_backend_end = datetime.now()

    logger.info("Posting results to backend took | %.3f secs.",
                time_diff(t_post_results_backend_start, t_post_results_backend_end))

    return posting_status

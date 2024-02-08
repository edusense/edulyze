"""
Author: Anonymized
Created: Anonymized

Common util to post https request to remote graphql endpoints
"""

import logging
import numpy as np
import pandas as pd
import traceback
from datetime import datetime
import base64
import requests
import json

# Import project level functions and classes
from configs.constants import Constants
from utils.time_utils import time_diff
from configs.graphql_query_schemas import build_query


class GraphQLDataFetchException(Exception):
    """Raised when request generates errors due to any reasons"""
    pass


def fetch_graphql_data(data_fetch_request, session_input_object, logger, credentials_prefix=""):
    """
    Post http requests to remote graphql request

    Parameters:
        data_fetch_request(string)     : request to post to graphql endpoint
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        logger(logger)                 : The logging object from parent
        credentials_prefix(String)     : In case we need different server endpoint, and credentials for audio

    Returns:
        response_data(dict)             : Dictionary of json data fetched from remote
    """


    t_post_graphql_request_start = datetime.now()
    response_data = None

    # prepare credentials and header to connect backend endpoint
    if credentials_prefix == "":
        frame_url = f'https://{session_input_object.get("server_backend_url")}/query'
        logger.debug(f'Frame url posted to {frame_url}')

        credential_string = '{}:{}'.format(
            session_input_object.get("server_user"),
            session_input_object.get("server_password"))
        credential_encoded = base64.standard_b64encode(credential_string.encode('ascii')).decode('ascii')

        request_headers = {'Authorization': ('Basic %s' % credential_encoded), 'Content-Type': 'application/json'}
    else:
        frame_url = f'https://{session_input_object.get(f"{credentials_prefix}server_backend_url")}/query'
        logger.debug(f'Frame url posted to {frame_url}')

        credential_string = '{}:{}'.format(
            session_input_object.get(f"{credentials_prefix}server_user"),
            session_input_object.get(f"{credentials_prefix}server_password"))
        credential_encoded = base64.standard_b64encode(credential_string.encode('ascii')).decode('ascii')

        request_headers = {'Authorization': ('Basic %s' % credential_encoded), 'Content-Type': 'application/json'}

    # post request, and format response
    try:
        raw_response = requests.post(frame_url, headers=request_headers, json=data_fetch_request)
    except ConnectionError:
        logger.error("Connection error for request")
        logger.error(str(traceback.print_exc()))
        raise GraphQLDataFetchException()

    response_json = raw_response.json()

    # Check if request is successfull across various levels

    if (not response_json.get("success")) | (not response_json.get("response")):
        logger.error("Request failed unexpectedly, bad request..")
        raise GraphQLDataFetchException(str(response_json))

    response_dict = json.loads(dict(response_json)["response"])

    if not response_dict.get("data"):
        logger.error("No data received in the request")
        raise GraphQLDataFetchException(str(response_json))
    else:
        response_data = response_dict.get("data")

    t_post_graphql_request_end = datetime.now()

    logger.info("Receiving raw response for Graphql query took | %.3f secs. ", time_diff(t_post_graphql_request_start,
                                                                                         t_post_graphql_request_end))
    return response_data


def sample_output_query(session_input_object):
    """ Sample query and way to call analytics schema"""

    # credentials for backend posting

    output_server_user = session_input_object.get('output_server_name')
    output_server_password = session_input_object.get('output_server_password')
    output_server_url = session_input_object.get('output_server_url')

    cred = '{}:{}'.format(output_server_user, output_server_password).encode('ascii')
    encoded_cred = base64.standard_b64encode(cred).decode('ascii')

    # Query template and posting results

    backend_params = {
        'headers': {
            'Authorization': 'Basic {}'.format(encoded_cred),
            'Content-Type': 'application/json'}
    }

    query = '''
        {
            analytics(sessionId: "608f3daadab4eb0918826dda", keyword: "cmu_05418A_ghc_4102_201905011200") {
                id
                keyword
                metaInfo {
                    pipelineVersion
                }
            }
        }
        '''
    req = {'query': query}
    resp = requests.post(output_server_url, headers=backend_params['headers'], json=req)

    print("****resp returned")
    if (resp.status_code != 200 or
            'success' not in resp.json().keys() or
            not resp.json()['success']):
        raise RuntimeError(resp.text)

    response = dict(resp.json())

    return response


class NoStudentDataError(Exception):
    pass


class NoInstructorDataError(Exception):
    pass

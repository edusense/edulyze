"""
Author: Anonymized
Created: Anonymized

This file contains schemas for various kinds of graphql queries to fetch data
"""

from datetime import datetime
from utils.time_utils import time_diff


class QueryBuilderException(Exception):
    """Raised when request query cannot be build"""
    pass


class graphql_query_set:
    """Schemas for different kinds of queries"""
    video_all_query = '''
        {{
            sessions {0}{{
                id
                videoFrames{1}{{
                    frameNumber
                    timestamp{{
                        RFC3339
                        unixSeconds
                        unixNanoseconds
                    }}
                    people{{
                        body
                        face
                        hand
                        openposeId
                        inference{{
                            posture{{
                                armPose
                                sitStand
                                centroidDelta
                                armDelta
                            }}
                            face{{
                                boundingBox
                                mouthOpen
                                mouthSmile
                                orientation
                            }}
                            head{{
                                roll
                                pitch
                                yaw
                                translationVector
                                gazeVector
                            }}
                            trackingId
                        }}
                    }} 
                }}
            }}
        }}
        '''

    audio_all_query = '''
        {{
            sessions{0}{{
                id
                audioFrames{1}{{
                    frameNumber
                    timestamp{{
                        RFC3339
                        unixSeconds
                        unixNanoseconds
                    }}
                    channel
                    audio{{
                        amplitude
                        melFrequency
                        mfccFeatures
                        polyFeatures
                    }}
                }}
            }}
        }}
        '''

    ## Extend it for application level example and smarter queries


def build_query(query_type, session_id=None, channel=None, schema='classinsight-graphql', logger=None):
    """
    Build query for corresponding query request

    Parameters:
        query_type(string)               : Type of query, audio or video, can be extend to other types
        session_id(string)               : If querying for particular session, then session id
        channel(string)                  : If querying for particular channel(instructor/student), then channel id
        schema(string)                   : backend schema type for given query

    Returns:
        query(string)                    : Final constructed query for request
    """

    t_build_query_start = datetime.now()

    if query_type == 'video':
        base_query = graphql_query_set.video_all_query
    elif query_type == 'audio':
        base_query = graphql_query_set.audio_all_query
    else:
        raise QueryBuilderException(f"query_type: {query_type} not supported..")

    if session_id is not None:
        if channel is not None:
            final_query = base_query.format(
                f'(sessionId: "{session_id}" )',
                f'(schema: "{schema}", channel : {channel})')
        else:
            final_query = base_query.format(
                f'(sessionId: "{session_id}" )',
                f'(schema: "{schema}")')
    elif channel is not None:
        final_query = base_query.format(
            '',
            f'(schema: "{schema}", channel : {channel})')
    else:
        final_query = base_query.format('', '')

    t_build_query_end = datetime.now()

    if logger is not None:
        logger.debug("Query Building took | %.3f secs ", time_diff(t_build_query_start, t_build_query_end))

    return final_query

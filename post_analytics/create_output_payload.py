"""
Author: Anonymized
Created: Anonymized
Create empty json output schema from session output object
"""
# Import python library functions
import json
import sys
import os
import logging
from datetime import datetime
from itertools import chain

# Import external library functions
import pandas as pd
import statistics
import math
import numpy as np

# Import project level functions and classes
from configs.constants import Constants, exitStatus
from utils.time_utils import time_diff
from post_analytics.init_empty_output_schema import init_empty_schema, get_empty_block_schema, \
    get_empty_second_schema, \
    get_empty_session_schema


def create_output_payload(session_input_object, session_output_object, logger_pass):
    """
    This is main wrapper function to create complete json payload

    Parameters:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        session_output_object(dict)    : Dictionary to collect all outputs for session analytics
        logger_pass(logger)            : The parent logger from which to derive the logger for this function

    Returns:
        output payload(dict)             : Final Payload in dict format
    """
    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('create_payload')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_create_payload_start = datetime.now()

    # initialize empty analytics schema and add id and keyword
    payload_dict = init_empty_schema()

    payload_dict['id'] = session_input_object.get("session_id")
    payload_dict['keyword'] = session_input_object.get("session_keyword")

    # --------meta_info--------

    meta_info = {
        'pipelineVersion': Constants.ANALYTICS_PIPELINE_VERSION,
        'analysisStartTime': session_input_object.get('analysis_start_time', -1),
        'totalRuntime': session_output_object.get('analysis_run_time', -1),
        'RunModules': session_output_object.get('run_modules'),
        'ModuleRuntime': session_output_object.get('module_runtimes'),
        'SuccessModules': session_output_object.get('success_modules'),
        # left general for now, need to make it strict like
        'FailureModules': session_output_object.get('failure_modules'),
        # Or('audio', 'gaze', 'location', 'posture', None)
    }

    # time info for blocks and seconds level
    df_video_time_input = session_input_object.get('input_location_df')[['timestamp', 'frameNumber', 'block_id']]
    df_video_time_input['second'] = (df_video_time_input['timestamp'] // Constants.MILLISECS_IN_SEC).astype(int)
    seconds_info_df = df_video_time_input.groupby('second').agg({'frameNumber': ['min', 'max']})
    seconds_info_df.columns = ['min_frame', 'max_frame']
    seconds_info_dict = seconds_info_df.to_dict(orient='index')
    block_info_dict = df_video_time_input.groupby('block_id')['second'].min().to_dict()

    # --------second level info--------

    # Get second level output df for audio

    audio_second_output = session_output_object.get('audio', {}).get('second')
    df_audio_second = None
    if len(audio_second_output.keys()) > 0:
        df_audio_second = pd.merge(
            audio_second_output['silence_seconds'],
            audio_second_output['object_noise_seconds'],
            on=['block_id', 'second'],
            how='outer'
        )
        for col in df_audio_second.columns:
            df_audio_second[col] = df_audio_second[col].astype(int)

    # Get second level output df for gaze

    df_gaze_instructor_second = session_output_object.get('gaze', {}).get('instructor', {}).get('second_level_info')
    df_gaze_student_second = session_output_object.get('gaze', {}).get('student', {}).get('second_level_info')

    # Get second level output df for location

    df_location_instructor_second = session_output_object.get('location', {}).get('instructor', {}).get(
        'second_level_info')
    df_location_student_second = session_output_object.get('location', {}).get('student', {}).get('second_level_info')

    # get overall seconds dataframe

    unique_seconds = pd.concat([
        df_audio_second[['second']] if df_audio_second is not None else pd.DataFrame(),
        df_location_instructor_second[['second']] if df_location_instructor_second is not None else pd.DataFrame(),
        df_location_student_second[['second']] if df_location_student_second is not None else pd.DataFrame(),
        df_gaze_instructor_second[['second']] if df_gaze_instructor_second is not None else pd.DataFrame(),
        df_gaze_student_second[['second']] if df_gaze_student_second is not None else pd.DataFrame()
    ]).second.drop_duplicates().sort_values()

    seconds_payload = []
    for second in unique_seconds.values:

        # Get empty schema

        sec_payload = get_empty_second_schema()

        # collect second inputs

        audio_sec = df_audio_second[df_audio_second.second == second].to_dict() if df_audio_second is not None else None
        location_instructor_sec = df_location_instructor_second[
            df_location_instructor_second.second == second].to_dict() if df_location_instructor_second is not None else None
        location_student_sec = df_location_student_second[
            df_location_student_second.second == second].to_dict() if df_location_student_second is not None else None
        gaze_instructor_sec = df_gaze_instructor_second[
            df_gaze_instructor_second.second == second].to_dict() if df_gaze_instructor_second is not None else None
        gaze_student_sec = df_gaze_student_second[
            df_gaze_student_second.second == second].to_dict() if df_gaze_student_second is not None else None

        # ---Info payload---
        info_payload = {
            'unixSeconds': int(second),
            'frameStart': seconds_info_dict.get(second, {}).get('min_frame'),
            'frameEnd': seconds_info_dict.get(second, {}).get('max_frame'),
        }
        sec_payload['secondInfo'].update(info_payload)

        # ---Audio payload---
        audio_idxs = sorted(list(audio_sec['second'].keys())) if audio_sec is not None else []
        if len(audio_idxs) > 0:
            audio_idx = audio_idxs[0]
            audio_payload = {
                'isSilence': bool(audio_sec['is_silence'][audio_idx]),
                'isObjectNoise': bool(audio_sec['is_object_noise'][audio_idx]),
                'isTeacherOnly': None,
                'isSingleSpeaker': None,
            }
            sec_payload['audio'].update(audio_payload)
        else:
            sec_payload['audio'] = None

        # ---Gaze Instructor payload---
        gaze_instructor_idxs = sorted(
            list(gaze_instructor_sec['second'].keys())) if gaze_instructor_sec is not None else []
        if len(gaze_instructor_idxs) > 0:
            gaze_instructor_idx = gaze_instructor_idxs[0]
            gaze_instructor_payload = {
                'angle': float(gaze_instructor_sec['gaze_dir'][gaze_instructor_idx]),
                'angleChange': None,
                'direction': gaze_instructor_sec['gaze_cat'][gaze_instructor_idx],
                'viewingSectorThreshold': None,
                'countStudentsInGaze': None,
                'towardsStudents': None,
                'lookingDown': None,
            }
            sec_payload['gaze']['instructor'].update(gaze_instructor_payload)
        else:
            sec_payload['gaze']['instructor'] = None

        # ---Location Instructor payload---
        location_instructor_idxs = sorted(
            list(location_instructor_sec['second'].keys())) if location_instructor_sec is not None else []
        if len(location_instructor_idxs) > 0:
            location_instructor_idx = location_instructor_idxs[0]
            location_instructor_payload = {
                'atBoard': bool(location_instructor_sec['at_board'][location_instructor_idx]),
                'atPodium': bool(location_instructor_sec['at_podium'][location_instructor_idx]),
                'isMoving': bool(location_instructor_sec['moving'][location_instructor_idx]),
                'locationCoordinates': [int(location_instructor_sec['loc_x'][location_instructor_idx]),
                                        int(location_instructor_sec['loc_y'][location_instructor_idx])],
                'locationCategory': location_instructor_sec['loc_cat'][location_instructor_idx],
                'locationEntropy': float(location_instructor_sec['body_entropy'][location_instructor_idx]),
                'headEntropy': float(location_instructor_sec['head_entropy'][location_instructor_idx]),
            }
            sec_payload['location']['instructor'].update(location_instructor_payload)
        else:
            sec_payload['location']['instructor'] = None

        # ---Gaze Student payload---
        gaze_student_idxs = sorted(list(gaze_student_sec['second'].keys())) if gaze_student_sec is not None else []
        if len(gaze_student_idxs) > 0:
            gaze_student_payload = {
                'id': [int(gaze_student_sec['trackingId'][idx]) for idx in gaze_student_idxs],
                'angle': [float(gaze_student_sec['gaze_dir'][idx]) for idx in gaze_student_idxs],
                'angleChange': None,
                'direction': [gaze_student_sec['gaze_cat'][idx] for idx in gaze_student_idxs],
                'towardsInstructor': None,
                'lookingDown': None,
                'lookingFront': [bool(gaze_student_sec['gaze_facing'][idx] == 'front') for idx in gaze_student_idxs],
            }
            sec_payload['gaze']['student'].update(gaze_student_payload)
        else:
            sec_payload['gaze']['student'] = None

        # ---location student payload
        location_student_idxs = sorted(
            list(location_student_sec['second'].keys())) if location_student_sec is not None else []
        if len(location_student_idxs) > 0:
            location_student_payload = {
                'id': [int(location_student_sec['trackingId'][idx]) for idx in location_student_idxs],
                'trackingIdMap': None,
                'isMoving': [bool(location_student_sec['is_moving'][idx]) for idx in location_student_idxs],
                'locationCoordinates': [
                    [int(location_student_sec['loc_x'][idx]), int(location_student_sec['loc_y'][idx])]
                    for idx in location_student_idxs],
                'locationCategory': [location_student_sec['loc_cat'][idx] for idx in location_student_idxs],
                'locationEntropy': [float(location_student_sec['body_entropy'][idx]) for idx in location_student_idxs],
                'headEntropy': [float(location_student_sec['head_entropy'][idx]) for idx in location_student_idxs],
            }
            sec_payload['location']['student'].update(location_student_payload)
        else:
            sec_payload['location']['student'] = None

        # set null payload for posture
        sec_payload['posture'] = None

        # Append second payload
        seconds_payload.append(sec_payload)

    # --------block level info--------

    audio_block_output = session_output_object.get('audio', {}).get('block')
    df_audio_block = None
    if len(audio_block_output.keys()) > 0:
        df_audio_block = pd.merge(
            audio_block_output['silence_block_fraction'],
            audio_block_output['object_noise_block_fraction'],
            on=['block_id'],
            how='outer',
            suffixes=('_silence', '_obj_noise')
        )
        df_audio_block['block_id'] = df_audio_block['block_id'].astype(int)
        df_audio_block.columns = ['block_id', 'silence_fraction', 'object_noise_fraction']

    # Get block level output df for gaze

    df_gaze_instructor_block = session_output_object.get('gaze', {}).get('instructor', {}).get('block_level_info')
    df_gaze_student_block = session_output_object.get('gaze', {}).get('student', {}).get('block_level_info')

    # Get block level output df for location

    df_location_instructor_block = session_output_object.get('location', {}).get('instructor', {}).get(
        'block_level_info')
    df_location_student_block = session_output_object.get('location', {}).get('student', {}).get('block_level_info')
    df_location_student_block_cluster = session_output_object.get('location', {}).get('student', {}).get(
        'block_cluster_info')

    # get unique blocks to dump into block payload

    unique_blocks = pd.concat([
        df_audio_block[['block_id']] if df_audio_block is not None else pd.DataFrame(),
        df_location_instructor_block[['block_id']] if df_location_instructor_second is not None else pd.DataFrame(),
        df_location_student_block[['block_id']] if df_location_student_second is not None else pd.DataFrame(),
        df_gaze_instructor_block[['block_id']] if df_gaze_instructor_second is not None else pd.DataFrame(),
        df_gaze_student_block[['block_id']] if df_gaze_student_second is not None else pd.DataFrame()
    ]).block_id.drop_duplicates().sort_values()

    blocks_payload = []
    for block_id in unique_blocks.values:
        # Get empty schema

        block_payload = get_empty_block_schema()

        # block inputs

        audio_block = df_audio_block[
            df_audio_block.block_id == block_id].to_dict() if df_audio_block is not None else None
        location_instructor_block = df_location_instructor_block[
            df_location_instructor_block.block_id == block_id].to_dict() if df_location_instructor_block is not None else None
        location_student_block = df_location_student_block[
            df_location_student_block.block_id == block_id].to_dict() if df_location_student_block is not None else None
        location_student_cluster_block = df_location_student_block_cluster[
            df_location_student_block_cluster.block_id == block_id].to_dict() if df_location_student_block_cluster is not None else None
        gaze_instructor_block = df_gaze_instructor_block[
            df_gaze_instructor_block.block_id == block_id].to_dict() if df_gaze_instructor_block is not None else None
        gaze_student_block = df_gaze_student_block[
            df_gaze_student_block.block_id == block_id].to_dict() if df_gaze_student_block is not None else None

        # ---Info payload---
        info_payload = {
            'unixStartSecond': block_info_dict.get(block_id, -1),
            'blockLength': Constants.BLOCK_SIZE,
            'id': int(block_id),
        }
        block_payload['blockInfo'].update(info_payload)

        # ---Audio payload---
        audio_idxs = sorted(list(audio_block['block_id'].keys())) if audio_block is not None else []
        if len(audio_idxs) > 0:
            audio_idx = audio_idxs[0]
            audio_payload = {
                'silenceFraction': float(audio_block['silence_fraction'][audio_idx]),
                'objectFraction': float(audio_block['object_noise_fraction'][audio_idx]),
                'teacherOnlyFraction': None,
                'singleSpeakerFraction': None,
                'teacherActivityType': None,
                'teacherActivityFraction': None,
                'teacherActivityTimes': None,
            }
            block_payload['audio'].update(audio_payload)
        else:
            block_payload['audio'] = None

        # ---Gaze Instructor payload---
        gaze_instructor_idxs = sorted(
            list(gaze_instructor_block['block_id'].keys())) if gaze_instructor_block is not None else []
        if len(gaze_instructor_idxs) > 0:
            gaze_instructor_idx = gaze_instructor_idxs[0]
            gaze_instructor_payload = {
                'gazeCategory': 'left' if gaze_instructor_block['IB_gaze_left'][gaze_instructor_idx] >
                                          gaze_instructor_block['IB_gaze_right'][gaze_instructor_idx] else 'right',
                'totalCategoryFraction': [float(gaze_instructor_block['IB_gaze_left'][gaze_instructor_idx]),
                                          float(gaze_instructor_block['IB_gaze_right'][gaze_instructor_idx])],
                'longestCategoryFraction': [float(gaze_instructor_block['IB_longest_left'][gaze_instructor_idx]),
                                            float(gaze_instructor_block['IB_longest_right'][gaze_instructor_idx])],
                'principalGaze': {
                    'direction': None,
                    'directionVariation': None,
                    'longestStayFraction': None,
                },
                'rollMean': None,
                'yawMean': None,
                'pitchMean': None,
                'rollVariance': None,
                'yawVariance': float(gaze_instructor_block['IB_yaw_variance'][gaze_instructor_idx]),
                'pitchVariance': None,
            }
            block_payload['gaze']['instructor'].update(gaze_instructor_payload)
        else:
            block_payload['gaze']['instructor'] = None

        # ---Location Instructor payload---
        location_instructor_idxs = sorted(
            list(location_instructor_block['block_id'].keys())) if location_instructor_block is not None else []
        if len(location_instructor_idxs) > 0:
            location_instructor_idx = location_instructor_idxs[0]
            location_instructor_payload = {
                'totalBoardFraction': float(location_instructor_block['IB_atboard'][location_instructor_idx]),
                'longestBoardFraction': float(location_instructor_block['IB_longest_board'][location_instructor_idx]),
                'totalPodiumFraction': float(location_instructor_block['IB_atpodium'][location_instructor_idx]),
                'longestPodiumFraction': float(location_instructor_block['IB_longest_podium'][location_instructor_idx]),
                'totalMovingFraction': float(location_instructor_block['IB_moving'][location_instructor_idx]),
                'longestMovingFraction': float(location_instructor_block['IB_longest_moving'][location_instructor_idx]),
                'locationCategory': ['left', 'right'],
                'CategoryFraction': [float(location_instructor_block['IB_class_left'][location_instructor_idx]),
                                     float(location_instructor_block['IB_class_right'][location_instructor_idx])],
                'longestCategoryFraction': [
                    float(location_instructor_block['IB_longest_class_left'][location_instructor_idx]),
                    float(location_instructor_block['IB_longest_class_right'][location_instructor_idx])],
                'stayAtLocation': [list(map(int, center)) for center in
                                   location_instructor_block['IB_clusters'][location_instructor_idx].tolist()],
                'stayAtLocationTimes': None,
                'longestStayFraction': float(location_instructor_block['IB_longest_still'][location_instructor_idx]),
                'principalMovement': {
                    'directionMean': None,
                    'directionVariation': None,
                    'directionComps': None,
                },
            }
            block_payload['location']['instructor'].update(location_instructor_payload)
        else:
            block_payload['location']['instructor'] = None

        # ---Gaze Student payload---
        gaze_student_idxs = sorted(
            list(gaze_student_block['block_id'].keys())) if gaze_student_block is not None else []
        if len(gaze_student_idxs) > 0:
            gaze_student_payload = {
                'id': [int(gaze_student_block['trackingId'][idx]) for idx in gaze_student_idxs],
                'numOccurrencesInBlock': [int(gaze_student_block['num_occurrences_in_block'][idx]) for idx in
                                          gaze_student_idxs],
                'gazeCategory': None,
                'totalCategoryFraction': [[float(gaze_student_block['SB_gaze_left'][idx]),
                                           float(gaze_student_block['SB_gaze_right'][idx])]
                                          for idx in gaze_student_idxs],
                'longestCategoryFraction': [[float(gaze_student_block['SB_longest_left'][idx]),
                                             float(gaze_student_block['SB_longest_right'][idx])]
                                            for idx in gaze_student_idxs],
                'directionMean': None,
                'directionVariation': None,
                'towardsInstructorFraction': None,
                'lookingDownFraction': None,
                'lookingFrontFraction': [float(gaze_student_block['SB_facing_front'][idx]) for idx in
                                         gaze_student_idxs],
                'rollMean': None,
                'yawMean': None,
                'pitchMean': None,
                'rollVariance': None,
                'yawVariance': [float(gaze_student_block['SB_yaw_variance'][idx]) for idx in
                                gaze_student_idxs],
                'pitchVariance': None,
            }
            block_payload['gaze']['student'].update(gaze_student_payload)
        else:
            block_payload['gaze']['student'] = None

        # ---location student payload
        location_student_idxs = sorted(
            list(location_student_block['block_id'].keys())) if location_student_block is not None else []
        location_cluster_idxs = sorted(list(
            location_student_cluster_block['block_id'].keys())) if location_student_cluster_block is not None else []
        if len(location_student_idxs) > 0:
            location_cluster_idx = location_cluster_idxs[0]
            location_student_payload = {
                'id': [int(location_student_block['trackingId'][idx]) for idx in location_student_idxs],
                'numOccurrencesInBlock': [int(location_student_block['num_occurences'][idx]) for idx in
                                          location_student_idxs],
                'isSettled': [bool(location_student_block['isSettled'][idx]) for idx in location_student_idxs],
                'meanBodyEntropy': [float(location_student_block['body_entropy_mean'][idx]) for idx in
                                    location_student_idxs],
                'maxBodyEntropy': [float(location_student_block['body_entropy_max'][idx]) for idx in
                                   location_student_idxs],
                'varBodyEntropy': [float(location_student_block['body_entropy_var'][idx]) for idx in
                                   location_student_idxs],
                'meanHeadEntropy': [float(location_student_block['head_entropy_mean'][idx]) for idx in
                                    location_student_idxs],
                'maxHeadEntropy': [float(location_student_block['head_entropy_max'][idx]) for idx in
                                   location_student_idxs],
                'varHeadEntropy': [float(location_student_block['head_entropy_var'][idx]) for idx in
                                   location_student_idxs],
                'stayCoordinates': [[int(location_student_block['loc_x'][idx]),
                                     int(location_student_block['loc_y'][idx])]
                                    for idx in location_student_idxs],
                'clusterCount': int(len(location_student_cluster_block['cluster_id'][location_cluster_idx])),
                'clusterCenters': [list(map(int, center)) for center in
                                   location_student_cluster_block['cluster_center'][location_cluster_idx]],
                'clusterStudentIds': [list(map(int, center)) for center in
                                      location_student_cluster_block['cluster_students'][location_cluster_idx]],
                'seatingArrangement': None,
            }
            block_payload['location']['student'].update(location_student_payload)
        else:
            block_payload['location']['student'] = None

        # set null payload for posture
        block_payload['posture'] = None

        # Append second payload
        blocks_payload.append(block_payload)

    # --------session level info--------

    # todo: add session level info based on block and seconds

    session_payload = None

    # Update final Payload dict
    payload_dict['metaInfo'].update(meta_info)
    if len(seconds_payload) > 0:
        payload_dict['secondLevel'] = seconds_payload
    if len(blocks_payload) > 0:
        payload_dict['blockLevel'] = blocks_payload
    if session_payload is not None:
        payload_dict['sessionLevel'] = session_payload

    t_create_payload_end = datetime.now()

    logger.info("Creating JSON payload took | %.3f secs.",
                time_diff(t_create_payload_start, t_create_payload_end))

    return payload_dict

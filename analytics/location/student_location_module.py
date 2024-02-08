"""
Author: Anonymized
Created: Anonymized

This file contains main student location functions
"""

# Import python library functions
import logging
from datetime import datetime

# Import external library functions
import pandas as pd

pd.options.mode.chained_assignment = None
import scipy.stats
import numpy as np

# Import project level functions and classes
from configs.constants import Constants
from utils.time_utils import time_diff
from analytics.location.utils import student_location_config
from analytics.location.utils import get_hip_location, get_shoulder_location, get_head_location
from analytics.location.utils import get_line_parameters, get_location_category
from analytics.location.get_student_clusters import get_student_clusters


def student_location_module(session_input_object, session_output_object, logger_pass):
    """
    This is main wrapper function to run location analytics for student particularly

    Parameters:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        session_output_object(dict)    : Dictionary to collect all outputs for session analytics
        logger_pass(logger)            : The parent logger from which to derive the logger for this function

    Returns:
        location_results(dict)         : Dictionary to collect all student location outputs
    """
    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('student_location')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_student_location_start = datetime.now()
    location_results = dict()

    # Get location config
    t_parse_input_start = datetime.now()
    location_config = session_input_object['session_meta_data'].get("location_config")
    if location_config is None:
        logger.error("Location config not available, not running any student location module")
        return location_results

    # Get input for entire module

    df_location_input = session_input_object.get('input_location_df')
    df_location_input = df_location_input[df_location_input.channel == 'student']

    if df_location_input.shape[0]==0:
        logger.error("No student location information present, skipping student location module")
        return location_results
    df_location_input = df_location_input.sort_values(by=['block_id', 'frameNumber', 'trackingId'])

    t_parse_input_end = datetime.now()
    logger.info("Parsing input for student location module took | %.3f secs.",
                time_diff(t_parse_input_start, t_parse_input_end))

    # Organise frame output and create location frame attributes
    t_get_coordinates_start =datetime.now()
    df_frame_output = df_location_input[
        ['timestamp', 'block_id', 'frameNumber', 'trackingId', 'body_kps', 'boundingBox']]

    # get student seating location per frame based on hip location

    df_frame_output.body_kps = df_frame_output.body_kps.apply(lambda x: list(map(int, x)))  # make all body kps ints

    df_frame_output['loc_x'] = None
    df_frame_output['loc_y'] = None

    for sid in df_frame_output.trackingId.unique():
        df_sid = df_frame_output[df_frame_output.trackingId == sid]
        df_sid['stu_loc'] = df_sid['body_kps'].apply(lambda x: get_hip_location(x))
        df_sid['cx'] = df_sid['stu_loc'].apply(lambda x: x[0])
        df_sid['cy'] = df_sid['stu_loc'].apply(lambda x: x[1])
        df_sid = df_sid.fillna(method='ffill').fillna(method='bfill').fillna(-1)
        df_frame_output.loc[df_frame_output.trackingId == sid, 'loc_x'] = df_sid['cx']
        df_frame_output.loc[df_frame_output.trackingId == sid, 'loc_y'] = df_sid['cy']

    df_frame_output.loc_x = df_frame_output.loc_x.astype(float)
    df_frame_output.loc_y = df_frame_output.loc_y.astype(float)

    # get location category of student based on center line(left/right)

    center_line = location_config.get('student', {}).get('center_line')
    if center_line is None:
        # assume center line to be from middle(and camera is positioned on middle of class
        center_line = [[Constants.FULL_RES_X_MAX / 2, 0], [Constants.FULL_RES_X_MAX / 2, Constants.FULL_RES_Y_MAX]]

    # get location category

    cline_slope, cline_intercept = get_line_parameters(center_line)
    if cline_slope == np.inf:
        df_frame_output['loc_cat'] = df_frame_output['loc_x'].apply(
            lambda x: 'left' if x < center_line[0][0] else 'right')
    else:
        df_frame_output['loc_cat'] = df_frame_output.apply(
            lambda row: get_location_category(row['loc_x'], row['loc_y'], cline_slope, cline_intercept), axis=1)

    t_get_coordinates_end = datetime.now()
    logger.info("Getting location coordinates and category took | %.3f secs.",
                time_diff(t_get_coordinates_start, t_get_coordinates_end))


    # get shoulder and head diff at frame level to calculate entropy as second and block level
    t_entropy_start = datetime.now()

    df_frame_output['shoulder_diff'] = None
    df_frame_output['head_diff'] = None

    for sid in df_frame_output.trackingId.unique():
        df_sid = df_frame_output[df_frame_output.trackingId == sid]
        df_sid['timediff'] = df_sid.timestamp.diff() / Constants.MILLISECS_IN_SEC

        # get shoulder, and head location

        df_sid['shoulder_loc'] = df_sid['body_kps'].apply(lambda x: get_shoulder_location(x))
        df_sid['shoulder_x'] = df_sid['shoulder_loc'].apply(lambda x: x[0])
        df_sid['shoulder_y'] = df_sid['shoulder_loc'].apply(lambda x: x[1])

        df_sid['head_loc'] = df_sid['boundingBox'].apply(lambda x: get_head_location(x))
        df_sid['head_x'] = df_sid['head_loc'].apply(lambda x: x[0])
        df_sid['head_y'] = df_sid['head_loc'].apply(lambda x: x[1])

        # fill nan positions to previous, and then later values
        df_sid = df_sid.fillna(method='ffill').fillna(method='bfill')
        # get shoulder distances

        df_sid['shoulder_xdiff'] = df_sid['shoulder_x'].diff()
        df_sid['shoulder_ydiff'] = df_sid['shoulder_y'].diff()
        df_sid['shoulder_dist'] = (df_sid['shoulder_ydiff'].pow(2) + df_sid['shoulder_xdiff'].pow(2)) / df_sid[
            'timediff'].pow(2)
        df_sid['shoulder_dist'] = df_sid['shoulder_dist'].pow(0.5)
        try:
            df_sid.loc[np.isnan(df_sid.shoulder_dist.values, casting='unsafe'), 'shoulder_dist'] = 0
            df_sid.loc[np.isinf(df_sid.shoulder_dist.values, casting='unsafe'), 'shoulder_dist'] = 0
        except:
            continue
        df_frame_output.loc[df_frame_output.trackingId == sid, 'shoulder_diff'] = df_sid['shoulder_dist']

        # get head distances

        df_sid['head_xdiff'] = df_sid['head_x'].diff()
        df_sid['head_ydiff'] = df_sid['head_y'].diff()
        df_sid['head_dist'] = (df_sid['head_ydiff'].pow(2) + df_sid['head_xdiff'].pow(2)) / df_sid['timediff'].pow(2)
        df_sid['head_dist'] = df_sid['head_dist'].pow(0.5)
        df_sid.loc[np.isnan(df_sid.head_dist.values), 'head_dist'] = 0
        df_sid.loc[np.isinf(df_sid.head_dist.values), 'head_dist'] = 0
        df_frame_output.loc[df_frame_output.trackingId == sid, 'head_diff'] = df_sid['head_dist']

    # Get second level output based on frame level output

    df_frame_output['second'] = (df_frame_output['timestamp'] // Constants.MILLISECS_IN_SEC).astype(int)

    df_second_output = df_frame_output.groupby(['second', 'block_id', 'trackingId'], as_index=False).agg({
        'loc_x': np.nanmean,
        'loc_y': np.nanmean,
        'loc_cat': lambda x: scipy.stats.mode(x).mode[0],
        'shoulder_diff': np.nansum,
        'head_diff': np.nansum,
    })

    df_second_output.columns = ['second', 'block_id', 'trackingId', 'loc_x', 'loc_y', 'loc_cat', 'body_entropy',
                                'head_entropy']

    df_second_output['is_moving'] = (df_second_output.body_entropy > 0) | (df_second_output.head_entropy > 0)

    # Get block level output

    df_block_output = df_second_output.groupby(['block_id', 'trackingId'], as_index=False).agg({
        'loc_x': np.nanmean,
        'loc_y': np.nanmean,
        'body_entropy': [lambda x: np.percentile(x, 99), np.nanmean, np.nanvar],
        'head_entropy': [lambda x: np.percentile(x, 99), np.nanmean, np.nanvar],
        'loc_cat': 'count',
    })
    df_block_output.columns = ["_".join(x) for x in df_block_output.columns.ravel()]
    df_block_output.columns = ['block_id', 'trackingId', 'loc_x', 'loc_y',
                               'body_entropy_max', 'body_entropy_mean', 'body_entropy_var',
                               'head_entropy_max', 'head_entropy_mean', 'head_entropy_var',
                               'num_occurences']

    df_block_output = df_block_output.fillna(0)
    t_entropy_end = datetime.now()
    logger.info("Getting entropy values took | %.3f secs.",
                time_diff(t_entropy_start, t_entropy_end))

    # get total frames in a block
    t_is_settled_start = datetime.now()
    df_frame_count = df_frame_output.groupby('block_id', as_index=False)['frameNumber'].nunique()
    df_frame_count.columns = ['block_id', 'frame_count']
    df_block_output = pd.merge(df_block_output, df_frame_count, on='block_id')

    # get isSettled for each student id

    presence_fraction_threshold = student_location_config.get('SETTLED_MIN_FRACTION')
    movement_var_qtile_threshold = student_location_config.get('SETTLED_MAX_VARIANCE_PERCENTILE')
    min_variance_to_include = student_location_config.get('SETTLED_MIN_VARIANCE')
    movement_var_threshold = df_second_output.loc[
        df_second_output.body_entropy > min_variance_to_include, 'body_entropy'].quantile(movement_var_qtile_threshold)

    df_block_output['presence_fraction'] = df_block_output.num_occurences / df_block_output.frame_count
    df_block_output['isSettled'] = (df_block_output['presence_fraction'] > presence_fraction_threshold) & \
                                   (df_block_output['body_entropy_max'] < movement_var_threshold)
    t_is_settled_end = datetime.now()
    logger.info("Getting student settlement took | %.3f secs.",
                time_diff(t_is_settled_start, t_is_settled_end))

    # get student clusters for each block

    cluster_output = []
    logger.info(f"Running clustering for all blocks")
    t_clustering_start = datetime.now()

    for bid in df_block_output.block_id.unique():
        df_bid = df_block_output[df_block_output.block_id == bid]
        df_bid = df_bid[df_bid.isSettled]
        # logger.info(f"Running clustering for block id | {str(bid)}")
        cluster_ids, cluster_centres, cluster_students = get_student_clusters(
            df_bid[['trackingId', 'loc_x', 'loc_y']],
            center_line, logger_pass)
        cluster_output.append([bid, cluster_ids, cluster_centres, cluster_students])

    df_cluster_output = pd.DataFrame(cluster_output,
                                     columns=['block_id', 'cluster_id', 'cluster_center', 'cluster_students'])
    t_clustering_end = datetime.now()
    logger.info("Student Clustering took | %.3f secs.",
                time_diff(t_clustering_start, t_clustering_end))
    # IsSettled at block level based on viewed student id counts

    df_tids_block = df_block_output[['block_id', 'trackingId']].drop_duplicates()
    df_tids_block[df_tids_block.trackingId < 0] = -1
    df_tids_block['exists'] = 1
    df_tids_block = pd.pivot_table(df_tids_block, index='block_id', columns='trackingId', values='exists',
                                   aggfunc=np.mean)

    BLOCK_SETTLED_MODE_PERCENTILE = student_location_config.get('BLOCK_SETTLED_MODE_PERCENTILE')
    is_settled_block = df_tids_block.sum(axis=1).reset_index()
    is_settled_block['settled'] = is_settled_block[0] >= scipy.stats.mode(is_settled_block[0].values).mode[
        0] * BLOCK_SETTLED_MODE_PERCENTILE
    is_settled_block.columns = ['block_id', 'positive_id_count', 'isSettledBlock']

    df_block_output = pd.merge(df_block_output, is_settled_block, on='block_id')
    # Running all student location modules complete

    location_results = {
        # 'frame_level_info': df_frame_output,
        'second_level_info': df_second_output,
        'block_level_info': df_block_output,
        'block_cluster_info': df_cluster_output
    }

    t_student_location_end = datetime.now()

    logger.info("Student Location Analysis took | %.3f secs.",
                time_diff(t_student_location_start, t_student_location_end))

    return location_results

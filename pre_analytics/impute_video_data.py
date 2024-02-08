"""
Author: Anonymized
Created: Anonymized

This file contains functions imputes missing video input data using various techniques.
"""
import logging
import traceback

import numpy as np
import pandas as pd
from datetime import datetime
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt

# Import project level functions and classes
from configs.constants import Constants
from utils.time_utils import time_diff
from pre_analytics.utils import get_bounding_box_distance, preops_config, get_edges, impute_tid_config, \
    interpolate_frame_info


def impute_video_data(session_input_object, processed_video_data, logger_pass):
    """
    Wrapper function to imputes various video features for processed video data

    Parameters:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        processed_video_data(dict)     : Dictionary containing video data processed for analytics engine
        logger_pass(logger)            : The parent logger from which to derive the logger for the engine

    Returns:
        imputed_video_data(dict)      : Dictionary containing video data with missing values imputed for analytics
        imputation_metrics(dict)      : Dictionary containing information about characterstics of imputation performed
    """
    # initialize logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('impute_video_data')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_impute_video_data_start = datetime.now()

    # Sync student tracking Ids

    synced_student_data, synced_student_id_metrics = sync_student_tracking_ids(processed_video_data['student'], logger)

    # Impute student tracking Ids

    imputed_student_data, imputed_student_id_metrics = impute_student_tracking_ids(synced_student_data, logger)

    # reset and fill teacher tracking Ids to one

    imputed_instructor_data, clean_instructor_id_metrics = clean_instructor_tracking_id(
        processed_video_data['instructor'], logger)

    # Todo: Add code to impute missing video data from ids

    imputed_video_data = {
        'student': imputed_student_data,
        'instructor': imputed_instructor_data
    }

    imputation_metrics = {
        'student_data_metrics': {
            'sync_id_metrics': synced_student_id_metrics,
            'impute_id_metrics': imputed_student_id_metrics,
        },
        'instructor_data_metrics': clean_instructor_id_metrics
    }

    t_impute_video_data_end = datetime.now()

    logger.info("Imputing video data took | %.3f secs. ",
                time_diff(t_impute_video_data_start, t_impute_video_data_end))

    return imputed_video_data, imputation_metrics


def impute_student_tracking_ids(student_data, logger):
    """
    Impute student ids across complete session

    Parameters:
        student_data(dict)             : Dictionary containing video data with synced tracking ids
        logger(logger)                 : Inherited logger from parent

    Returns:
        imputed_video_data(dict)       : Dictionary containing video data with imputed tracking ids
        impute_metrics(dict)           : Dictionary mapping additional information about imputing tracking ids
    """

    t_impute_tracking_ids_start = datetime.now()

    # Initialize synced student data structures

    quality_metrics = {
        'student_ids': [],
        'imputed_frames': [],
    }

    # get sid fid matrix by dataframing, melting and then pivoting dataset
    sid_fid_list = {frameNumber: list(student_data[frameNumber]['people'].keys())
                    for frameNumber in student_data.keys()}
    sid_fid_melted_df = pd.DataFrame.from_dict(sid_fid_list, orient='index').reset_index().melt(
        id_vars='index').sort_values(by=['index'])

    sid_fid_melted_df = sid_fid_melted_df[sid_fid_melted_df.value > 0][['index', 'value']].astype(int)
    sid_fid_melted_df.columns = ['frameNumber', 'trackingId']
    sid_fid_melted_df['exists'] = 1
    sid_fid_matrix = pd.pivot_table(sid_fid_melted_df.sort_values(by=['frameNumber', 'trackingId']),
                                    index='frameNumber', columns='trackingId', values='exists',
                                    aggfunc=np.sum, fill_value=0.)

    # get threshold values from config

    edge_length_threshold = impute_tid_config.get('EDGE_LENGTH_THRESHOLD')
    edge_max_gap_threshold = impute_tid_config.get('EDGE_MAX_GAP_THRESHOLD')
    edge_min_gap_threshold = impute_tid_config.get('EDGE_MIN_GAP_THRESHOLD')
    num_frames = sid_fid_matrix.shape[0]

    # Loop over all tracking Ids to impute values

    for col_idx, trackingId in enumerate(sid_fid_matrix.columns):
        edge_start, edge_end = get_edges(sid_fid_matrix[trackingId].values == 0)
        edge_length = edge_end - edge_start
        edge_gaps = edge_start[1:] - edge_end[:-1]
        if len(edge_length) ==0:
            continue
        top_edge_gaps = np.insert(edge_gaps, 0, edge_start[0])
        bottom_edge_gaps = np.insert(edge_gaps, edge_gaps.shape[0], num_frames - edge_end[-1])
        # check where edge length is less than some threshold

        fill_edges_idx = np.where(
            (edge_length <= edge_length_threshold) &
            # (np.maximum(top_edge_gaps, bottom_edge_gaps) >= edge_max_gap_threshold) &
            (np.minimum(top_edge_gaps, bottom_edge_gaps) >= edge_min_gap_threshold)
        )[0]

        for edge_idx in fill_edges_idx:  # impute frames for fill edges
            sid_fid_matrix.iloc[edge_start[edge_idx]:edge_end[edge_idx], col_idx] = 1.

            # Get previous frame
            prev_frame_number = sid_fid_matrix.index[edge_start[edge_idx] - 1]
            next_frame_number = sid_fid_matrix.index[edge_end[edge_idx]]

            impute_frames = interpolate_frame_info(
                student_data[prev_frame_number]['people'][trackingId],
                student_data[next_frame_number]['people'][trackingId],
                edge_length[edge_idx])

            impute_frame_ids = sid_fid_matrix.index[edge_start[edge_idx]:edge_end[edge_idx]].values
            for idx, frameNumber in enumerate(impute_frame_ids):
                student_data[frameNumber]['people'][trackingId] = impute_frames[idx]
                quality_metrics['student_ids'].append(trackingId)
                quality_metrics['imputed_frames'].append(frameNumber)

    t_impute_tracking_ids_end = datetime.now()

    logger.info("Imputing tracking ids took | %.3f secs.",
                time_diff(t_impute_tracking_ids_start, t_impute_tracking_ids_end))

    return student_data, quality_metrics


def sync_student_tracking_ids(student_data, logger):
    """
    Sync student ids across complete session

    Parameters:
        processed_video_data(dict)     : Dictionary containing video data processed for analytics engine
        logger(logger)                 : Inherited logger from parent

    Returns:
        synced_video_data(dict)       : Dictionary containing video data with synced tracking ids
        syncing_metrics(dict)         : Dictionary mapping additional information about syncing tracking ids
    """

    t_syncing_tracking_ids_start = datetime.now()
    # Initialize synced student data structures

    synced_student_data = dict()
    tracking_id_map = dict()
    unique_tracking_ids = None
    previous_student_frame = None
    previous_frame_ids = None

    # Stores how many frames of what type have we seen
    quality_metrics = {
        'no_new_ids': 0,
        'no_unmapped_new_ids': 0,
        'no_unmapped_prev_ids': 0,
        'frame_level_info': dict(),

    }

    # Loop over frames to and sync tracking ids with previous frames
    for frameNumber, student_frame in student_data.items():

        quality_metrics['frame_level_info'][frameNumber] = dict()

        # List all unique trackingIds in frame:
        frame_tracking_ids = list(student_frame['people'].keys())

        if unique_tracking_ids is None:  # Execute this when we see first frame
            unique_tracking_ids = deepcopy(frame_tracking_ids)
            synced_student_data[frameNumber] = student_frame

            previous_student_frame = student_frame
            previous_frame_ids = list(student_frame['people'].keys())
            logger.debug("First frame passed..")
            quality_metrics['frame_level_info'][frameNumber]['count_original_ids'] = deepcopy(unique_tracking_ids)
            continue

        # Check which of frame ids we do not see in previous frames

        new_frame_ids = [ids for ids in frame_tracking_ids if ids not in unique_tracking_ids]

        if len(new_frame_ids) == 0:  # If no new ids are there in this frame
            previous_student_frame = student_frame
            previous_frame_ids = frame_tracking_ids
            quality_metrics['no_new_ids'] += 1
            continue

        # see if we have mapping for any new_frame_ids, and replace them in student_data
        synced_student_data[frameNumber] = student_frame

        not_mapped_new_ids = list()

        for new_id in new_frame_ids:
            if new_id in tracking_id_map.keys():
                synced_student_data[frameNumber]['people'][tracking_id_map[new_id]] = \
                    synced_student_data[frameNumber]['people'][new_id]
                del synced_student_data[frameNumber]['people'][new_id]
            else:
                not_mapped_new_ids.append(new_id)

        if len(not_mapped_new_ids) < 0:  # if no non mapped ids are left
            previous_student_frame = synced_student_data[frameNumber]
            previous_frame_ids = list(synced_student_data[frameNumber]['people'].keys())
            logger.debug("All new ids mapped to old ids in this frame")
            quality_metrics['no_unmapped_new_ids'] += 1
            continue

        # Check if new id is suspiciously close to any of the older bounding boxes from previous frame

        non_mapped_prev_frame_ids = [sid for sid in previous_frame_ids if sid not in frame_tracking_ids]

        if len(non_mapped_prev_frame_ids) == 0:  # if no non mapped prev frame ids are left
            previous_student_frame = synced_student_data[frameNumber]
            previous_frame_ids = list(synced_student_data[frameNumber]['people'].keys())
            unique_tracking_ids += not_mapped_new_ids  # Adding all left over ids to unique ids.
            quality_metrics['no_unmapped_prev_ids'] += 1
            quality_metrics['frame_level_info'][frameNumber]['count_original_ids'] = len(not_mapped_new_ids)
            continue

        dummy_face_bb = [[0, 0], [0, 0]]  # for face bounding boxes not available
        old_ids_face_bbs = [previous_student_frame['people'][sid]['boundingBox'] for sid in non_mapped_prev_frame_ids]

        old_ids_face_bbs = [(old_id_face_bb if old_id_face_bb is not None else dummy_face_bb) for old_id_face_bb in
                            old_ids_face_bbs]

        old_faces_bb_size = np.array([np.min(np.absolute(np.array(old_id_face_bb[0]) - np.array(old_id_face_bb[1]))) / 2
                                      for old_id_face_bb in old_ids_face_bbs if old_id_face_bb])

        is_old_id_mapped = np.zeros_like(non_mapped_prev_frame_ids)

        independent_new_ids = []
        none_id_distance_threshold = preops_config.get('none_id_distance_threshold')
        face_not_present_count = 0
        face_id_not_present_count = 0
        for new_id in not_mapped_new_ids:
            new_id_face_bb = synced_student_data[frameNumber]['people'][new_id]['boundingBox']

            if new_id_face_bb is None:  # face bb not present
                if (new_id > 0):  # only include non faces id if they were not none. else not include them at all
                    face_not_present_count += 1
                    independent_new_ids.append((new_id, -1))
                else:
                    face_id_not_present_count += 1
                    del synced_student_data[frameNumber]['people'][new_id]
                continue

            new_face_bb_size = np.min(np.absolute(np.array(new_id_face_bb[0]) - np.array(new_id_face_bb[1]))) / 2

            dist_old_ids = np.array([get_bounding_box_distance(new_id_face_bb, old_id_face_bb) for old_id_face_bb in
                                     old_ids_face_bbs])

            dist_old_ids[np.where(is_old_id_mapped)[0]] = np.inf  # set already mapped ids distance to infinity

            bb_box_overlap = dist_old_ids - np.maximum(old_faces_bb_size, new_face_bb_size)
            closest_prev_frame_id_idx = np.argmin(bb_box_overlap)

            if (bb_box_overlap[closest_prev_frame_id_idx] < 0) & (  # i.e both bounding box overlaps
                    old_faces_bb_size[closest_prev_frame_id_idx] > 0):  # older one is not dummy
                # Todo: Make this threshold based, to make sure we do not merge unnecessary ids
                tracking_id_map[new_id] = non_mapped_prev_frame_ids[closest_prev_frame_id_idx]
                is_old_id_mapped[closest_prev_frame_id_idx] = 1

                # update this frame id in synced frame data
                synced_student_data[frameNumber]['people'][non_mapped_prev_frame_ids[closest_prev_frame_id_idx]] = \
                    synced_student_data[frameNumber]['people'][new_id]
                tracking_id_map[new_id] = non_mapped_prev_frame_ids[closest_prev_frame_id_idx]
                del synced_student_data[frameNumber]['people'][new_id]
            elif (new_id < 0) & (bb_box_overlap[closest_prev_frame_id_idx] < none_id_distance_threshold):
                tracking_id_map[new_id] = non_mapped_prev_frame_ids[closest_prev_frame_id_idx]
                is_old_id_mapped[closest_prev_frame_id_idx] = 1

                # update this frame id in synced frame data
                synced_student_data[frameNumber]['people'][non_mapped_prev_frame_ids[closest_prev_frame_id_idx]] = \
                    synced_student_data[frameNumber]['people'][new_id]
                tracking_id_map[new_id] = non_mapped_prev_frame_ids[closest_prev_frame_id_idx]
                del synced_student_data[frameNumber]['people'][new_id]
            else:
                independent_new_ids.append((new_id, round(bb_box_overlap[closest_prev_frame_id_idx], 2)))

        quality_metrics['frame_level_info'][frameNumber]['face_not_present_count'] = face_not_present_count
        quality_metrics['frame_level_info'][frameNumber]['face_id_not_present_count'] = face_id_not_present_count

        if len(independent_new_ids) > 0:  # if there are any independent/distant ids left
            quality_metrics['frame_level_info'][frameNumber]['count_original_ids'] = len(independent_new_ids)
            quality_metrics['frame_level_info'][frameNumber]['original_id_dist_map'] = independent_new_ids
            unique_tracking_ids += [id[0] for id in independent_new_ids]

        previous_student_frame = synced_student_data[frameNumber]
        previous_frame_ids = list(synced_student_data[frameNumber]['people'].keys())

    syncing_ids_metrics = {
        'unique_tracking_ids': unique_tracking_ids,
        'tracking_id_map': tracking_id_map,
        'tracking_id_quality_metrics': quality_metrics
    }

    t_syncing_tracking_ids_end = datetime.now()

    logger.info("Syncing tracking ids took | %.3f secs.",
                time_diff(t_syncing_tracking_ids_start, t_syncing_tracking_ids_end))

    return synced_student_data, syncing_ids_metrics


def clean_instructor_tracking_id(instructor_data, logger):
    """
    Reset all instructor ids to one, and drop any unknown ids

    Parameters:
        processed_instructor_data(dict): Dictionary containing instructor video data processed for analytics engine
        logger(logger)                 : Inherited logger from parent

    Returns:
        cleaned_video_data(dict)       : Dictionary containing video data with cleaned tracking ids
        cleaning_metrics(dict)         : Dictionary mapping additional information about cleaning tracking ids
    """

    t_clean_instructor_tracking_id_start = datetime.now()
    cleaned_instructor_data = dict()
    cleaning_metrics = {
        'frames_ids_removed': list(),
        'frames_ids_changed': list(),
        'frames_ids_added': list(),
        'prev_frames_not_available': list()
    }

    prev_instructor_frame = None
    INSTRUCTOR_TRACKING_ID = preops_config.get('instructor_tracking_id', 1)

    # Loop over frames to and clean/add/remove tracking ids
    for frameNumber, instructor_frame in instructor_data.items():

        frame_tracking_ids = list(instructor_frame['people'].keys())
        cleaned_instructor_data[frameNumber] = instructor_frame

        if (len(frame_tracking_ids) == 1):

            # for all frame with one person, set id to 1 forcefully
            cleaned_instructor_data[frameNumber]['people'][INSTRUCTOR_TRACKING_ID] = \
                deepcopy(cleaned_instructor_data[frameNumber]['people'][frame_tracking_ids[0]])

            if not (frame_tracking_ids[0] == INSTRUCTOR_TRACKING_ID):
                del cleaned_instructor_data[frameNumber]['people'][frame_tracking_ids[0]]
            prev_instructor_frame = cleaned_instructor_data[frameNumber]
            cleaning_metrics['frames_ids_changed'].append(frameNumber)

        elif len(frame_tracking_ids) == 0:

            # for frames with no person information copy previous frame information
            if prev_instructor_frame is not None:
                try:
                    cleaned_instructor_data[frameNumber]['people'][INSTRUCTOR_TRACKING_ID] = \
                        deepcopy(prev_instructor_frame['people'][INSTRUCTOR_TRACKING_ID])
                except:
                    logger.error(traceback.format_exc())

                cleaning_metrics['frames_ids_added'].append(frameNumber)
            else:
                cleaning_metrics['prev_frames_not_available'].append(frameNumber)

        else:  # len frame_tracking_ids > 1

            # for multiple frames, pick first one of them, and set it to one, remove all other frames
            if INSTRUCTOR_TRACKING_ID not in frame_tracking_ids:
                selected_id = frame_tracking_ids[0]

                cleaned_instructor_data[frameNumber]['people'][INSTRUCTOR_TRACKING_ID] = \
                    deepcopy(cleaned_instructor_data[frameNumber]['people'][selected_id])
                for id in frame_tracking_ids[1:]:
                    del cleaned_instructor_data[frameNumber]['people'][id]
            else:
                for id in frame_tracking_ids:
                    if not (id == INSTRUCTOR_TRACKING_ID):
                        del cleaned_instructor_data[frameNumber]['people'][id]

            prev_instructor_frame = cleaned_instructor_data[frameNumber]
            cleaning_metrics['frames_ids_removed'].append(frameNumber)

    t_clean_instructor_tracking_id_end = datetime.now()

    logger.info("Cleaning instructor tracking ids took | %.3f secs.",
                time_diff(t_clean_instructor_tracking_id_start, t_clean_instructor_tracking_id_end))

    return cleaned_instructor_data, cleaning_metrics


def location_based_sync_student_ids(df_frame_id_locations, logger):
    """
    Reset all instructor ids to one, and drop any unknown ids

    Parameters:
        df_frame_id_locations(dict)    : Dataframe with frame level locations for given tracking Ids
        logger(logger)                 : Inherited logger from parent

    Returns:
        df_frame_id_synced(dict)       : Dataframe with new set of ids
        id_sync_metrics(dict)          : Dictionary mapping additional information about syncing tracking ids
    """

    df_frame_id_synced = df_frame_id_locations.copy(deep=True)
    syncing_metrics = dict()

    # create

    return df_frame_id_synced


"""
Author: Anonymized
Created: Anonymized

This file contains code to detect object noise in classes at second level, and create statistics at block level.
"""

# Import python library functions
import logging
from datetime import datetime
import numpy as np
from copy import deepcopy
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing

# Import project level functions and classes
from configs.constants import Constants, exitStatus
from utils.time_utils import time_diff
from analytics.audio.utils import audio_config


def run_object_noise_detection_module(session_input_object, session_output_object, logger_pass):
    """
    This is main function to run object noise detection modules

    Parameters:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        session_output_object(dict)    : Dictionary to collect all outputs for session analytics
        logger_pass(logger)            : The parent logger from which to derive the logger for this function

    Returns:
        session_output_object(dict)    : Dictionary to collect all outputs for session analytics
    """
    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('object_noise_detection_module')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_object_noise_detection_start = datetime.now()

    # Initialize object noise detection module

    object_noise_detection_results = {
        'second': dict(),
        'block': dict(),
        'session': dict(),
        'debug': dict(),
    }

    run_status = exitStatus.SUCCESS

    # Extract MFCC and poly features from audio input
    audio_input_df = session_input_object.get('input_audio_df', None)
    if audio_input_df is not None:
        audio_input = audio_input_df[audio_input_df.channel=='instructor'].to_dict(orient='records')
        audio_input = {xr['frameNumber']:xr for xr in audio_input}
    else:
        t_object_noise_detection_end = datetime.now()
        run_status = exitStatus.NO_AUDIO_DATA
        logger.info("Unable to run Object noise detection module | %.3f secs.",
                    time_diff(t_object_noise_detection_start, t_object_noise_detection_end))

        return object_noise_detection_results, run_status

    # audio_input = session_input_object.get('audio', {}).get('instructor')

    mfcc_input = [np.array(audio_input[frameNumber]['mfccFeatures']) for frameNumber in audio_input.keys()]
    poly_input = [np.array(audio_input[frameNumber]['polyFeatures']) for frameNumber in audio_input.keys()]
    block_ids = np.array(
        [np.full(mfcc_input[i].shape[1], audio_input[frameNumber]['block_id']) for i, frameNumber in
         enumerate(audio_input.keys())]).flatten()
    timestamp_vals = np.array(
        [np.full(mfcc_input[i].shape[1], audio_input[frameNumber]['timestamp']) for i, frameNumber in
         enumerate(audio_input.keys())]).flatten()
    second_vals = timestamp_vals // Constants.MILLISECS_IN_SEC
    is_object_noise = np.zeros_like(second_vals)

    if mfcc_input[0] is None:
        logger.error("MFCC input not available, exiting module..")
        return object_noise_detection_results, exitStatus.FAILURE

    # Create complete session mfcc and poly features and do min/max scaling for them

    session_mfcc = np.concatenate(mfcc_input, axis=1)
    mfcc_min_max_scaler = preprocessing.MinMaxScaler()
    session_mfcc_scaled = mfcc_min_max_scaler.fit_transform(session_mfcc.T).T

    session_poly = np.concatenate(poly_input, axis=1)
    poly_min_max_scaler = preprocessing.MinMaxScaler()
    session_poly_scaled = poly_min_max_scaler.fit_transform(session_poly.T).T

    # start object detection object from silence mask from detected silences
    df_object_noise_detection = deepcopy(session_output_object['audio'].get('second', {}).get('silence_seconds'))

    if df_object_noise_detection is None:
        logger.error("Silence mask not available, exiting module..")
        return object_noise_detection_results, exitStatus.FAILURE

    # Append second level mfcc and poly feature to object detection df

    df_mfcc = pd.DataFrame(session_mfcc_scaled.T)
    df_mfcc['second'] = second_vals
    df_mfcc = df_mfcc.groupby('second')[df_mfcc.columns[:-1]].median()
    df_mfcc.columns = [f"mfcc_{col}" for col in df_mfcc.columns]

    df_poly = pd.DataFrame(session_poly_scaled.T)
    df_poly['second'] = second_vals
    df_poly = df_poly.groupby('second')[df_poly.columns[:-1]].median()
    df_poly.columns = [f"poly_{col}" for col in df_poly.columns]

    df_object_noise_detection = pd.merge(df_object_noise_detection, df_poly, on='second')
    df_object_noise_detection = pd.merge(df_object_noise_detection, df_mfcc, on='second')

    # loop over all blocks to detect object noises
    df_object_noise_detection['is_object_noise'] = 0.

    for block_id in range(np.max(block_ids) + 1):
        df_block_obj_detection = df_object_noise_detection[df_object_noise_detection.block_id == block_id]
        if df_block_obj_detection.shape[0] > 0:
            block_object_noise = detect_object_noise(df_block_obj_detection, logger)
            df_object_noise_detection.loc[df_object_noise_detection.block_id == block_id, 'is_object_noise'] = block_object_noise

    # Fill second, block and session level results

    object_noise_detection_results['debug']['object_noise_detection_df'] = df_object_noise_detection

    df_second_object_noise_detection = df_object_noise_detection.groupby(['block_id', 'second'], as_index=False)[
        'is_object_noise'].agg(pd.Series.mode)
    object_noise_detection_results['second']['object_noise_seconds'] = df_second_object_noise_detection

    df_block_fraction = df_object_noise_detection.groupby(['block_id'], as_index=False).agg(
        {'is_object_noise': ['count', 'sum']})
    df_block_fraction['block_fraction'] = df_block_fraction['is_object_noise']['sum'] / \
                                          df_block_fraction['is_object_noise']['count']
    object_noise_detection_results['block']['object_noise_block_fraction'] = df_block_fraction[
        ['block_id', 'block_fraction']]

    session_fraction = df_object_noise_detection['is_object_noise'].sum() / df_object_noise_detection.shape[0]
    object_noise_detection_results['session']['object_noise_session_fraction'] = session_fraction

    t_object_noise_detection_end = datetime.now()

    logger.info("Object noise detection module took | %.3f secs.",
                time_diff(t_object_noise_detection_start, t_object_noise_detection_end))

    return object_noise_detection_results, run_status


def detect_object_noise(df_object_noise_detection, logger, debug_plot=False):
    """
    This function detects object noises at second level based on MFCC/poly features

    Parameters:
        df_object_noise_detection(pd.DataFrame)   : context with silence information for object noise detection
        logger(logger)                      : The parent logger
        debug_plot(bool)                    : Whether we run it in debug mode or not

    Returns:
        is_object_noise(np.ndarray)         : An array marking each second as object noise/ not object noise
    """

    # Initialize input from audio config

    NORM_DIFF_THRESHOLD = audio_config.get('NORM_DIFF_THRESHOLD')
    OBJECT_NOISE_FEATURES = audio_config.get('OBJECT_NOISE_FEATURES')
    NUM_CLUSTERS = 2  # this is not configurable

    # create X vector from selected features
    feature_columns = []
    for mfcc_id in OBJECT_NOISE_FEATURES['mfcc']:
        feature_columns.append(f"mfcc_{mfcc_id}")

    for poly_id in OBJECT_NOISE_FEATURES['poly']:
        feature_columns.append(f"poly_{poly_id}")

    X_features = df_object_noise_detection[feature_columns].values

    # remove silence seconds from X_vec
    silence_mask = df_object_noise_detection['is_silence'].values
    non_silence_idx = np.where(silence_mask == 0)[0]
    X_features = X_features[non_silence_idx]

    # Exit if not completely silent block or only one point in block
    if X_features.shape[0] <= 1:
        logger.debug("Silent Block, return no object noise detected..")
        return np.zeros(df_object_noise_detection.shape[0])

    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0).fit(X_features)

    # determine if difference between cluster centers is sufficient for object noise to be present

    processed_kmeans_labels = kmeans.labels_
    max_norm = max(np.linalg.norm(kmeans.cluster_centers_[0]), np.linalg.norm(kmeans.cluster_centers_[1]))
    rel_norm_diff = np.linalg.norm(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1]) / max_norm

    is_object_noise = np.full(df_object_noise_detection.shape[0], fill_value=-1)

    if rel_norm_diff > NORM_DIFF_THRESHOLD:
        # Set all object noise seconds to 1, and rest to 0

        object_noise_label = np.argmax(np.linalg.norm(kmeans.cluster_centers_, axis=1))
        is_object_noise[non_silence_idx] = processed_kmeans_labels
        is_object_noise[~(is_object_noise == object_noise_label)] = -1
        is_object_noise[(is_object_noise == object_noise_label)] = 1
        is_object_noise[(is_object_noise == -1)] = 0

    else:
        # Difference is not sufficient, return no object noise detected
        is_object_noise[:] = 0
        pass

    return is_object_noise

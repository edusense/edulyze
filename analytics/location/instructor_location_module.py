"""
Author: Anonymized
Created: Anonymized

This file contains main instructor location functions
"""

# Import python library functions
import logging
from datetime import datetime
from itertools import chain

# Import external library functions
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import statistics
import scipy.stats
import numpy as np

# Import project level functions and classes
from configs.constants import Constants
from utils.time_utils import time_diff
from analytics.location.utils import perc_of_list, body_entropy, head_entropy, longest_consecutive, at_loc, \
    class_side, moving
from analytics.location.utils import get_instructor_movement

def instructor_location_module(session_input_object, session_output_object, logger_pass):
    """
    The instructor_location_module function is the main wrapper function to run location analytics for instructor
        particularly.

    Args:
        session_input_object: Pass the input dataframe for location analytics
        session_output_object: Collect all the outputs from this function
        logger_pass: Pass the logger from parent to child functions

    Returns:
        A dictionary containing the following keys:

    Doc Author:
        Trelent
    """
    """
    The instructor_location_module function is the main wrapper function to run location analytics for instructor
        particularly.
    
    Args:
        session_input_object: Pass the input dataframe
        session_output_object: Store the output of this function
        logger_pass: Pass the parent logger to this function
    
    Returns:
        A dictionary with the following keys:
    
    Doc Author:
        Trelent
    """
    """
    This is main wrapper function to run location analytics for instructor particularly

    Parameters:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        session_output_object(dict)    : Dictionary to collect all outputs for session analytics
        logger_pass(logger)            : The parent logger from which to derive the logger for this function

    Returns:
        location_results(dict)         : Dictionary to collect all instructor location outputs
    """
    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('instructor_location')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_instructor_location_start = datetime.now()

    # Get location config
    location_config = session_input_object['session_meta_data'].get("location_config")
    if location_config is None:
        logger.error("Location config not available, not running any instructor location module")
        return session_output_object
    stand_still_radius = session_input_object['session_meta_data'].get('stand_still_radius',10)

    # Run all modules

    # todo: All function written in dev manner, need to change it to production manner

    full_df = session_input_object['input_location_df']
    ins_df = full_df[full_df["channel"] == "instructor"]

    # run instructor movement module
    df_instructor_movement = get_instructor_movement(ins_df, stand_still_radius)
    ins_df['timestamp'] = ins_df['timestamp'].astype(np.int64) // 10**6
    df_instructor_movement = df_instructor_movement[df_instructor_movement.type=='stop']
    df_instructor_movement['timestamp'] = df_instructor_movement['timestamp'].astype(np.int64) // 10**6
    df_instructor_movement['last'] = pd.to_datetime(df_instructor_movement['last'], format='%Y-%m-%d %H:%M:%S').astype(np.int64) // 10**6
    instructor_stops = df_instructor_movement[['timestamp','last']].values.tolist()

    instructor_block_dfs = []
    instructor_second_dfs = []
    # *** GET INSTRUCTOR FRAME DATA ***
    for unique_id in ins_df.trackingId.unique():
        ins_df_id = ins_df[ins_df.trackingId==unique_id].sort_values(by='timestamp')
        # features for instructor frame df
        IF_timestamp = []
        IF_frames = []
        IF_at_podium = []
        IF_at_board = []
        IF_moving = []
        IF_moving_old = []
        IF_class_side = []
        IF_body_entropy = []
        IF_head_entropy = []
        IF_coords = []
        IF_block = []

        # iterate through instructor df and save data

        for i in range(len(ins_df_id)):
            # create row
            row = ins_df_id.iloc[i]
            IF_timestamp.append(row["timestamp"])
            IF_frames.append(row["frameNumber"])
            IF_at_podium.append(at_loc(row, location_config['instructor']['podiums']))
            IF_at_board.append(at_loc(row, location_config['instructor']['boards']))
            flattened_center_line_coor = list(chain.from_iterable(location_config['instructor']['center_line']))
            IF_class_side.append(class_side(row, flattened_center_line_coor))
            IF_body_entropy.append(body_entropy(row))
            IF_block.append(row["block_id"])

            if i == 0:
                IF_moving.append(False)
                IF_moving_old.append(False)
                IF_head_entropy.append(0)
            else:
                is_stop = False
                for (stop_start, stop_end) in instructor_stops:
                    if stop_start <= row['timestamp'] <= stop_end:
                        is_stop=True
                IF_moving_old.append(moving(ins_df_id.iloc[i - 1], ins_df_id.iloc[i], stand_still_radius))
                IF_moving.append(not is_stop)
                IF_head_entropy.append(head_entropy(ins_df_id.iloc[i - 1], ins_df_id.iloc[i]))

            if row["gazeVector"] is not None:
                IF_coords.append(row["gazeVector"][0])
            elif row["loc"] is not None:
                IF_coords.append(row["loc"][0:2])
            else:
                IF_coords.append([0, 0])

        # add frame data to df
        IF_df = pd.DataFrame()
        IF_df["IF_timestamp"] = IF_timestamp
        IF_df["IF_frames"] = IF_frames
        IF_df["IF_at_podium"] = IF_at_podium
        IF_df["IF_at_board"] = IF_at_board
        IF_df["IF_moving"] = IF_moving
        IF_df["IF_moving_old"] = IF_moving_old
        IF_df["IF_class_side"] = IF_class_side
        IF_df["IF_body_entropy"] = IF_body_entropy
        IF_df["IF_head_entropy"] = IF_head_entropy
        IF_df["IF_coords"] = IF_coords
        IF_df["block_id"] = IF_block

        # *** GET INSTRUCTOR BLOCK DATA ***

        # features for instructor block data
        IB_df = pd.DataFrame()
        blocks = IF_df["block_id"].unique()
        IB_atpodium = []
        IB_atboard = []
        IB_longest_podium = []
        IB_longest_board = []
        IB_moving = []
        IB_moving_old = []
        IB_longest_moving = []
        IB_longest_moving_old = []
        IB_longest_still = []
        IB_longest_still_old = []
        IB_class_left = []
        IB_class_right = []
        IB_longest_class_left = []
        IB_longest_class_right = []
        IB_principal_movement = []
        IB_clusters = []
        IB_head_entropy = []
        IB_body_entropy = []

        # iterate blocks and save data
        for block in blocks:
            bl_df = IF_df[IF_df["block_id"] == block]
            IB_atpodium.append(perc_of_list(bl_df["IF_at_podium"], True))
            IB_atboard.append(perc_of_list(bl_df["IF_at_board"], True))
            IB_moving.append(perc_of_list(bl_df["IF_moving"], True))
            IB_moving_old.append(perc_of_list(bl_df["IF_moving_old"], True))
            IB_class_left.append(perc_of_list(bl_df["IF_class_side"], "left"))
            IB_class_right.append(perc_of_list(bl_df["IF_class_side"], "right"))

            IB_longest_class_left.append(longest_consecutive(bl_df["IF_class_side"], "left"))
            IB_longest_class_right.append(longest_consecutive(bl_df["IF_class_side"], "right"))
            IB_longest_board.append(longest_consecutive(bl_df["IF_at_board"], True))
            IB_longest_podium.append(longest_consecutive(bl_df["IF_at_podium"], True))
            IB_longest_moving.append(longest_consecutive(bl_df["IF_moving"], True))
            IB_longest_moving_old.append(longest_consecutive(bl_df["IF_moving_old"], True))
            IB_longest_still.append(longest_consecutive(bl_df["IF_moving"], False))
            IB_longest_still_old.append(longest_consecutive(bl_df["IF_moving_old"], False))

            IB_head_entropy.append(statistics.mean(bl_df["IF_head_entropy"]))
            IB_body_entropy.append(statistics.mean(bl_df["IF_body_entropy"]))

            points = bl_df["IF_coords"]
            X = np.array(list(points))
            if X.shape[0] > 1:
                kmeans = KMeans(n_clusters=min(X.shape[0],5)).fit(X)
                IB_clusters.append(kmeans.cluster_centers_)
                pca = PCA(n_components=2)
                pca.fit(X)
                IB_principal_movement.append({
                    "mean": pca.mean_,
                    "vars": pca.explained_variance_,
                    "comps": pca.components_
                })
            else:
                IB_clusters.append(np.array([]))
                IB_principal_movement.append({
                    "mean": None,
                    "vars": None,
                    "comps": None
                })


            # print(block)

        # add block data to df
        IB_df["block_id"] = list(blocks)
        IB_df["IB_atpodium"] = IB_atpodium
        IB_df["IB_atboard"] = IB_atboard
        IB_df["IB_longest_podium"] = IB_longest_podium
        IB_df["IB_longest_board"] = IB_longest_board
        IB_df["IB_moving"] = IB_moving
        IB_df["IB_moving_old"] = IB_moving_old
        IB_df["IB_longest_moving"] = IB_longest_moving
        IB_df["IB_longest_moving_old"] = IB_longest_moving_old
        IB_df["IB_longest_still"] = IB_longest_still
        IB_df["IB_longest_still_old"] = IB_longest_still_old
        IB_df["IB_class_left"] = IB_class_left
        IB_df["IB_class_right"] = IB_class_right
        IB_df["IB_longest_class_left"] = IB_longest_class_left
        IB_df["IB_longest_class_right"] = IB_longest_class_right
        IB_df["IB_principal_movement"] = IB_principal_movement
        IB_df["IB_clusters"] = IB_clusters
        IB_df["IB_head_entropy"] = IB_head_entropy
        IB_df["IB_body_entropy"] = IB_body_entropy


        # Get second level results

        IF_df['second'] = (IF_df['IF_timestamp'] // Constants.MILLISECS_IN_SEC).astype(int)

        IS_df = IF_df.groupby(['second', 'block_id'], as_index=False).agg({
            'IF_at_podium': lambda x: scipy.stats.mode(x).mode[0],
            'IF_at_board': lambda x: scipy.stats.mode(x).mode[0],
            'IF_moving': lambda x: scipy.stats.mode(x).mode[0],
            'IF_moving_old': lambda x: scipy.stats.mode(x).mode[0],
            'IF_class_side': lambda x: scipy.stats.mode(x).mode[0],
            'IF_body_entropy': np.nansum,
            'IF_head_entropy':np.nansum,
            'IF_coords': [lambda x: np.mean([xr[0] for xr in x]),
                          lambda x: np.mean([xr[1] for xr in x])]
        })

        IS_df.columns = ['second', 'block_id', 'at_podium', 'at_board', 'moving','moving_old','loc_cat','body_entropy','head_entropy','loc_x','loc_y']
        instructor_second_dfs.append(IS_df)
        instructor_block_dfs.append(IB_df)



    location_results = {
        # 'frame_level_info': IF_df,
        'second_level_info': pd.concat(instructor_second_dfs),
        'block_level_info': pd.concat(instructor_block_dfs),
        'instructor_movement':df_instructor_movement
    }

    t_instructor_location_end = datetime.now()

    logger.info("Instructor Location Analysis took | %.3f secs.",
                time_diff(t_instructor_location_start, t_instructor_location_end))

    return location_results

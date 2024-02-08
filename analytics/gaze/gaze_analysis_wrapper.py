"""
Author: Anonymized
Created: Anonymized

This file contains wrapper function to run end to end analysis on gaze data
"""

# Import python library functions
import logging
from datetime import datetime
from itertools import chain
import scipy.stats

# Import external library functions
import pandas as pd
import statistics
import math
import numpy as np

# Import project level functions and classes
from configs.constants import Constants
from utils.time_utils import time_diff
from analytics.gaze.utils import gaze_config


def gaze_analysis_wrapper(session_input_object, session_output_object, logger_pass):
    """
    This is main wrapper function to run all gaze modules

    Parameters:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        session_output_object(dict)    : Dictionary to collect all outputs for session analytics
        logger_pass(logger)            : The parent logger from which to derive the logger for this function

    Returns:
        session_output_object(dict)    : Dictionary to collect all outputs for session analytics
    """
    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('gaze_analysis')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_gaze_analysis_start = datetime.now()

    # Get location config
    location_config = session_input_object['session_meta_data'].get("location_config")
    if location_config is None:
        logger.error("Location config not available, not running any instructor module")
        return session_output_object

    instructor_center_line_flat = list(chain.from_iterable(location_config['instructor']['center_line']))
    student_center_line_flat = list(chain.from_iterable(location_config['student']['center_line']))

    full_df = session_input_object['input_gaze_df']
    if full_df is None:
        logger.error("Gaze information not available, not running any gaze module")
        return session_output_object

    ins_df = full_df[full_df["channel"] == "instructor"]
    stu_df = full_df[full_df["channel"] == "student"]

    # Get data from gaze config for looking down angle
    instructor_lookdown_angle_threshold = gaze_config.get('INSTRUCTOR_LOOKDOWN_THRESHOLD_DEGREE')
    instructor_lookdown_quantile = gaze_config.get('INSTRUCTOR_LOOKDOWN_QUANTILE')
    student_lookdown_angle_threshold = gaze_config.get('STUDENT_LOOKDOWN_THRESHOLD_DEGREE')
    student_lookdown_quantile = gaze_config.get('STUDENT_LOOKDOWN_QUANTILE')

    # Get pitch limit for looking down for instructor
    instructor_lookdown_angle = ins_df['pitch'].fillna(np.nanmedian(ins_df['pitch'])).quantile(instructor_lookdown_quantile) - instructor_lookdown_angle_threshold


    instructor_block_dfs = []
    instructor_second_dfs = []
    # *** GET INSTRUCTOR FRAME DATA ***
    for unique_id in ins_df.trackingId.unique():
        ins_df_id = ins_df[ins_df.trackingId==unique_id].sort_values(by='timestamp')
        # features for instructor frame data df
        IF_timestamp = []
        IF_frames = []
        IF_block_id = []
        IF_gaze_dir = []
        IF_gaze_side = []
        IF_gaze_same_side = []
        IF_gaze_down = []
        IF_gaze_pitch = []
        IF_gaze_roll = []
        IF_armpose = []
        IF_sitstand = []
        IF_gaze_facing = []

        # iterate instructor df and save frame data
        for i in range(len(ins_df_id)):
            row = ins_df_id.iloc[i]

            IF_timestamp.append(row['timestamp'])
            IF_frames.append(row["frameNumber"])
            IF_block_id.append(row["block_id"])
            IF_gaze_side.append(gaze_dir(row, instructor_center_line_flat))
            IF_gaze_down.append(row['pitch'] <= instructor_lookdown_angle)
            IF_gaze_facing.append(row['orientation'])
            IF_armpose.append(row['armPose'])
            IF_sitstand.append(row['sitStand'])

            if row["yaw"] is None:
                IF_gaze_dir.append(0)
            else:
                IF_gaze_dir.append(row["yaw"])

            if row["pitch"] is None:
                IF_gaze_pitch.append(0)
            else:
                IF_gaze_pitch.append(row["pitch"])

            if row["roll"] is None:
                IF_gaze_roll.append(0)
            else:
                IF_gaze_roll.append(row["roll"])

            if i == 0:
                IF_gaze_same_side.append(False)
            else:
                IF_gaze_same_side.append(gaze_angle_same(ins_df_id.iloc[i - 1], ins_df_id.iloc[i]))

        # add frame data to df
        IF_df = pd.DataFrame()
        IF_df["IF_timestamp"] = IF_timestamp
        IF_df["IF_frames"] = IF_frames
        IF_df["IF_gaze_dir"] = IF_gaze_dir
        IF_df["IF_gaze_same_dir"] = IF_gaze_same_side
        IF_df["IF_gaze_side"] = IF_gaze_side
        IF_df["block_id"] = IF_block_id
        IF_df["IF_gaze_down"] = IF_gaze_down
        IF_df["IF_gaze_pitch"] = IF_gaze_pitch
        IF_df["IF_gaze_roll"] = IF_gaze_roll
        IF_df["IF_gaze_facing"] = IF_gaze_facing
        IF_df["IF_armpose"] = IF_armpose
        IF_df["IF_sitstand"] = IF_sitstand


        # *** GET INSTRUCTOR BLOCK DATA ***

        # features for instructor block df
        IB_df = pd.DataFrame()
        blocks = IF_df["block_id"].unique()
        IB_gaze_left = []
        IB_gaze_right = []
        IB_gaze_down = []
        IB_samedir = []
        IB_longest_samedir = []
        IB_longest_left = []
        IB_longest_right = []
        IB_yaw_variance = []
        IB_facing_front = []
        IB_facing_back = []

        # iterate blocks and save data
        for block in blocks:
            bl_df = IF_df[IF_df["block_id"] == block]

            IB_gaze_left.append(perc_of_list(bl_df["IF_gaze_side"], "left"))
            IB_gaze_right.append(perc_of_list(bl_df["IF_gaze_side"], "right"))
            IB_samedir.append(perc_of_list(bl_df["IF_gaze_same_dir"], True))
            IB_gaze_down.append(perc_of_list(bl_df["IF_gaze_down"], True))

            IB_facing_back.append(perc_of_list(bl_df["IF_gaze_facing"], "back"))
            IB_facing_front.append(perc_of_list(bl_df["IF_gaze_facing"], "front"))

            IB_longest_left.append(longest_consecutive(bl_df["IF_gaze_side"], "left"))
            IB_longest_right.append(longest_consecutive(bl_df["IF_gaze_side"], "right"))
            IB_longest_samedir.append(longest_consecutive(bl_df["IF_gaze_same_dir"], True))

            if len(list(bl_df["IF_gaze_dir"])) > 1:
                IB_yaw_variance.append(statistics.variance(list(bl_df["IF_gaze_dir"])))
            else:
                IB_yaw_variance.append(0)

        # add block data to df
        IB_df["block_id"] = list(blocks)
        IB_df["IB_gaze_left"] = IB_gaze_left
        IB_df["IB_gaze_right"] = IB_gaze_right
        IB_df["IB_same_dir"] = IB_samedir
        IB_df["IB_longest_left"] = IB_longest_left
        IB_df["IB_longest_right"] = IB_longest_right
        IB_df["IB_longest_samedir"] = IB_longest_samedir
        IB_df["IB_yaw_variance"] = IB_yaw_variance
        IB_df["IB_gaze_down"] = IB_gaze_down
        IB_df["IB_facing_back"] = IB_facing_back
        IB_df["IB_facing_front"] = IB_facing_front

        # Add results at seconds level for instructors

        IF_df['second'] = (IF_df['IF_timestamp'] // Constants.MILLISECS_IN_SEC).astype(int)

        IS_df = IF_df.groupby(['second', 'block_id'], as_index=False).agg({
            'IF_gaze_dir': np.nanmean,
            'IF_gaze_same_dir': np.any,
            'IF_gaze_side': lambda x: scipy.stats.mode(x).mode[0],
            'IF_gaze_down': lambda x: scipy.stats.mode(x).mode[0],
            'IF_gaze_pitch': np.nanmean,
            'IF_gaze_roll': np.nanmean,
            'IF_gaze_facing': lambda x: scipy.stats.mode(x).mode[0],
            'IF_armpose': lambda x: scipy.stats.mode([xr for xr in x if not (xr == 'error')] + ['z']).mode[0],
            'IF_sitstand': lambda x: scipy.stats.mode([xr for xr in x if not (xr == 'error')] + ['z']).mode[0],

        })

        IS_df.columns = ['second', 'block_id', 'gaze_dir', 'gaze_same_dir', 'gaze_cat', 'gaze_down', 'gaze_pitch',
                         'gaze_roll', 'gaze_facing', 'armpose', 'sitstand']

        instructor_second_dfs.append(IS_df)
        instructor_block_dfs.append(IB_df)


    # *** GET STUDENT FRAME DATA ***

    # add student looking down angle based on individual pitch value
    df_student_lookingdown_angle = stu_df.groupby('trackingId', as_index=False).agg(
        {'pitch': lambda x: x.quantile(student_lookdown_quantile) - student_lookdown_angle_threshold})
    df_student_lookingdown_angle.columns = ['trackingId', 'lookdown_angle']
    student_lookingdown_angle_dict = df_student_lookingdown_angle.fillna('ffill').set_index("trackingId").to_dict('index')

    # features for student frame df
    SF_timestamp = []
    SF_frames = []
    SF_person_id = []
    SF_block_id = []
    SF_gaze_dir = []
    SF_gaze_side = []
    SF_gaze_facing = []
    SF_gaze_down = []
    SF_gaze_pitch = []
    SF_gaze_roll = []
    SF_armpose = []
    SF_sitstand = []

    # iterate student frames df and save data
    for i in range(len(stu_df)):
        row = stu_df.iloc[i]
        SF_timestamp.append(row["timestamp"])
        SF_frames.append(row["frameNumber"])
        SF_block_id.append(row["block_id"])
        SF_person_id.append(row["trackingId"])
        SF_gaze_side.append(gaze_dir(row, student_center_line_flat))
        SF_gaze_facing.append(row["orientation"])
        SF_armpose.append(row['armPose'])
        SF_sitstand.append(row['sitStand'])

        student_lookdown_angle = student_lookingdown_angle_dict.get(row["trackingId"],{}).get('lookdown_angle',0)
        SF_gaze_down.append(row["pitch"] <= student_lookdown_angle)

        if row["yaw"] is None:
            SF_gaze_dir.append(0)
        else:
            SF_gaze_dir.append(row["yaw"])

        if row["pitch"] is None:
            SF_gaze_pitch.append(0)
        else:
            SF_gaze_pitch.append(row["pitch"])

        if row["roll"] is None:
            SF_gaze_roll.append(0)
        else:
            SF_gaze_roll.append(row["roll"])

    # add frame data to df
    SF_df = pd.DataFrame()
    SF_df["SF_timestamp"] = SF_timestamp
    SF_df["SF_frames"] = SF_frames
    SF_df["trackingId"] = SF_person_id
    SF_df["SF_gaze_dir"] = SF_gaze_dir
    SF_df["SF_gaze_side"] = SF_gaze_side
    SF_df["SF_gaze_facing"] = SF_gaze_facing
    SF_df["block_id"] = SF_block_id
    SF_df["SF_gaze_down"] = SF_gaze_down
    SF_df["SF_gaze_pitch"] = SF_gaze_pitch
    SF_df["SF_gaze_roll"] = SF_gaze_roll
    SF_df["SF_armpose"] = SF_armpose
    SF_df["SF_sitstand"] = SF_sitstand


    # *** GET STUDENT BLOCK DATA ***

    # features for student block df
    students = SF_df["trackingId"].unique()
    SB_df = pd.DataFrame()
    SB_blocks = []
    SB_ids = []
    SB_gaze_left = []
    SB_gaze_right = []
    SB_facing_front = []
    SB_facing_back = []
    SB_longest_left = []
    SB_longest_right = []
    SB_yaw_variance = []
    SB_num_per_block = []
    SB_gaze_down = []

    # iterate students in list of unique students
    for student in students:
        id = student
        ind_df = SF_df[SF_df["trackingId"] == id]
        blocks = ind_df["block_id"].unique()

        # get blocks that selected student is in, run block analyses
        for block in blocks:
            bl_df = ind_df[ind_df["block_id"] == block]

            SB_blocks.append(block)
            SB_ids.append(id)
            SB_num_per_block.append(len(bl_df))

            SB_gaze_left.append(perc_of_list(bl_df["SF_gaze_side"], "left"))
            SB_gaze_right.append(perc_of_list(bl_df["SF_gaze_side"], "right"))

            SB_facing_back.append(perc_of_list(bl_df["SF_gaze_facing"], "back"))
            SB_facing_front.append(perc_of_list(bl_df["SF_gaze_facing"], "front"))
            SB_gaze_down.append(perc_of_list(bl_df["SF_gaze_down"], True))

            if len(list(bl_df["SF_gaze_dir"])) > 1:
                SB_yaw_variance.append(statistics.variance(list(bl_df["SF_gaze_dir"])))
            else:
                SB_yaw_variance.append(0)

            SB_longest_left.append(longest_consecutive(bl_df["SF_gaze_side"], "left"))
            SB_longest_right.append(longest_consecutive(bl_df["SF_gaze_side"], "right"))

    #  save block data to df
    SB_df["block_id"] = SB_blocks
    SB_df["trackingId"] = SB_ids
    SB_df["num_occurrences_in_block"] = SB_num_per_block
    SB_df["SB_gaze_left"] = SB_gaze_left
    SB_df["SB_gaze_right"] = SB_gaze_right
    SB_df["SB_longest_left"] = SB_longest_left
    SB_df["SB_longest_right"] = SB_longest_right
    SB_df["SB_facing_back"] = SB_facing_back
    SB_df["SB_facing_front"] = SB_facing_front
    SB_df["SB_yaw_variance"] = SB_yaw_variance
    SB_df["SB_gaze_down"] = SB_gaze_down


    # Add results at seconds level for students and instructors

    SF_df['second'] = (SF_df['SF_timestamp'] // Constants.MILLISECS_IN_SEC).astype(int)

    SS_df = SF_df.groupby(['second', 'block_id', 'trackingId'], as_index=False).agg({
        'SF_gaze_dir': np.nanmean,
        'SF_gaze_side': lambda x: scipy.stats.mode(x).mode[0],
        'SF_gaze_facing': lambda x: scipy.stats.mode(x).mode[0],
        'SF_gaze_down': lambda x: scipy.stats.mode(x).mode[0],
        'SF_gaze_pitch': np.nanmean,
        'SF_gaze_roll': np.nanmean,
        'SF_armpose': lambda x: scipy.stats.mode([xr for xr in x if not (xr == 'error')] + ['z']).mode[0],
        'SF_sitstand': lambda x: scipy.stats.mode([xr for xr in x if not (xr == 'error')] + ['z']).mode[0],

    })

    SS_df.columns = ['second', 'block_id', 'trackingId', 'gaze_dir', 'gaze_cat', 'gaze_facing','gaze_down','gaze_pitch','gaze_roll','armpose','sitstand']

    # Append gaze results into session output object

    session_output_object['gaze'] = {
        'instructor': {
            # 'frame_level_info': IF_df,
            'second_level_info': pd.concat(instructor_second_dfs),
            'block_level_info': pd.concat(instructor_block_dfs),
        },
        'student': {
            # 'frame_level_info': SF_df,
            'second_level_info': SS_df,
            'block_level_info': SB_df,
        }
    }

    t_gaze_analysis_end = datetime.now()

    logger.info("Gaze Analysis took | %.3f secs.",
                time_diff(t_gaze_analysis_start, t_gaze_analysis_end))

    return session_output_object


def perc_of_list(ls, val):
    return list(ls).count(val) / len(ls)


def longest_consecutive(ls, val):
    max = 0
    temp_max = 0
    for item in ls:
        if item == val:
            temp_max += 1
        else:
            if temp_max > max:
                max = temp_max
            temp_max = 0
    if temp_max > max:
        max = temp_max

    return max / len(ls)


# determine where instructor gaze directed
def gaze_dir(row, div):
    if row["gazeVector"] is None or row["yaw"] is None:
        return "none"

    y1, x1 = row["gazeVector"][0][0], row["gazeVector"][0][1]
    line_length = 2000
    angle = row["yaw"]
    (y2, x2) = (y1 + line_length * math.cos(math.radians(angle)), x1 + line_length * math.sin(math.radians(angle)) * -1)

    return (place_target(div[0], div[1], div[2], div[3], x2, y2))


# helper: gaze_dir
def place_target(x1, y1, x2, y2, p_x, p_y):
    v1 = (x2 - x1, y2 - y1)  # Vector 1
    v2 = (x2 - p_x, y2 - p_y)  # Vector 1
    xp = v1[0] * v2[1] - v1[1] * v2[0]  # Cross product

    if x2 != x1 and (y2 - y1) / (x2 - x1) > 0:
        if xp < 0:
            return "left"
        else:
            return "right"

    else:
        if xp < 0:
            return "right"
        else:
            return "left"


def gaze_angle_same(row1, row2):
    if row1["yaw"] is None or row2["yaw"] is None:
        return 0

    angle1 = row1["yaw"]
    angle2 = row2["yaw"]

    return abs(angle2 - angle1) < 5

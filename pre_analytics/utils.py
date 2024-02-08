"""
Author: Anonymized
Created: Anonymized

This file contains utils functions for pre-analytics-ops
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from copy import deepcopy

# Import project level functions and classes
from configs.constants import Constants


preops_config = {
    'none_id_distance_threshold': 100,
    'instructor_tracking_id' : 1
}

impute_tid_config = {
    'EDGE_LENGTH_THRESHOLD': 10,
    'EDGE_MAX_GAP_THRESHOLD': 5,
    'EDGE_MIN_GAP_THRESHOLD': 1
}

def time_diff(t_start, t_end):
    """
    Get time diff in secs

    Parameters:
        t_start(datetime)               : Start time
        t_end(datetime)                 : End time

    Returns:
        t_diff(int)                     : time difference in secs
    """

    return (t_end - t_start).seconds + np.round((t_end - t_start).microseconds / 1000000, 3)



def get_edges(arr):
    """ Return edges depicted by 1's in a boolean array"""
    if arr.sum() == 0:  # no edges
        return np.array([]), np.array([])

    if arr.sum() == arr.shape[0]:  # full edge
        return np.array([0]), np.array([arr.shape[0] - 1])

    # all other cases

    edge_start_idx = np.where(arr[1:] > arr[:-1])[0] + 1
    edge_end_idx = np.where(arr[1:] < arr[:-1])[0] + 1
    if arr[0] == 1:
        edge_start_idx = np.insert(edge_start_idx, 0, 0)

    if arr[-1] == 1:
        edge_end_idx = np.insert(edge_end_idx, edge_end_idx.shape[0], arr.shape[0])

    return edge_start_idx, edge_end_idx




def get_bounding_box_distance(bounding_box_1, bounding_box_2):
    """
    get distance between two bounding boxes in a video frame
    """

    bb1_center = np.array([np.mean([bounding_box_1[0][0], bounding_box_1[1][0]]),
                           np.mean([bounding_box_1[0][1], bounding_box_1[1][1]])])

    bb2_center = np.array([np.mean([bounding_box_2[0][0], bounding_box_2[1][0]]),
                           np.mean([bounding_box_2[0][1], bounding_box_2[1][1]])])

    return np.linalg.norm(bb1_center - bb2_center)


def interpolate_frame_info(start_frame, end_frame, num_steps):
    """
    Interpolate and create frame information based on start and end frames
    """

    interpolated_frames = [dict()]*num_steps

    interpolate_keys = ['body_kps','boundingBox','roll','pitch','yaw','translationVector','gazeVector']

    for frame_attr in start_frame.keys():
        if frame_attr in interpolate_keys:
            try:
                interpolated_attrs = np.linspace(start_frame[frame_attr], end_frame[frame_attr], num_steps+2)
                interpolated_attrs = interpolated_attrs[1:-1]
                for i in range(num_steps):
                    interpolated_frames[i][frame_attr] = interpolated_attrs[i]
            except:
                for i in range(num_steps):
                    interpolated_frames[i][frame_attr] = start_frame[frame_attr]
        else:
            for i in range(num_steps):
                interpolated_frames[i][frame_attr] = start_frame[frame_attr]

    return interpolated_frames
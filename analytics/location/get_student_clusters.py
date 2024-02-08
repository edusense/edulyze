"""
Author: Anonymized
Created: Anonymized

This file contains students clustering module at block level
"""

# Import python library functions
import logging
from datetime import datetime

# Import external library functions
import pandas as pd

pd.options.mode.chained_assignment = None
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np

# Import project level functions and classes
from configs.constants import Constants
from utils.time_utils import time_diff
from analytics.location.utils import student_location_config


def get_student_clusters(df_input, center_line, logger_pass):
    """
    This is student clustering module at block level

    Parameters:
        df_input(pd.DataFrame)         : Id and location information for clustering
        center_line(np.ndarray)        : Information about center line in room based on camera position from location config
        logger_pass(logger)            : The parent logger from which to derive the logger for this function

    Returns:
        cluster_ids(list)              : Ids of formed clusters
        cluster_centers(list)          : centres of formed clusters
        cluster_students(list)         : students in formed clusters
    """

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('student_cluster')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_student_cluster_start = datetime.now()

    # get transformed coordinates

    # df_cluster_input = get_transformed_coordinates(df_input, center_line)

    # init clustering output variables
    cluster_ids, cluster_centers, cluster_students = [], [], []

    # return in case we do not have any data
    if df_input.shape[0]==0:
        t_student_cluster_end = datetime.now()
        logger.warning("No settled students to find student cluster, exiting clustering in | %.3f secs.",
                    time_diff(t_student_cluster_start, t_student_cluster_end))
        return cluster_ids, cluster_centers, cluster_students

    # get data for clustering

    # todo: transformation closed, need to open it to test and compare
    # X = df_cluster_input[['cx', 'cy']].values
    # ids = df_cluster_input['trackingId'].values

    X = df_input[['loc_x', 'loc_y']].values
    ids = df_input['trackingId'].values

    # get SILHOUETTE score for all possible cluster count

    MIN_SILHOUETTE_SCORE = student_location_config.get('MIN_SILHOUETTE_SCORE')
    silhouette_scores = [MIN_SILHOUETTE_SCORE]
    for i in range(2, X.shape[0]):
        km = KMeans(n_clusters=i, random_state=0).fit(X)
        predictions = km.predict(X)
        silhouette = silhouette_score(X, predictions)
        silhouette_scores.append(silhouette)
        # logger.debug("Silhouette score for number of cluster(s) {}: {}".format(i, silhouette))

    # get cluster count with maximum silhouette score (returns 1 if all scores are low, i.e not clustering required)
    optimal_cluster_count = np.argmax(silhouette_scores) + 1

    # get final cluster_ids, centers and students in each cluster

    student_kmeans = KMeans(n_clusters=optimal_cluster_count).fit(X)

    for i in range(optimal_cluster_count):
        cluster_ids.append(i)
        cluster_centers.append(student_kmeans.cluster_centers_[i].tolist())
        cluster_students.append(ids[student_kmeans.labels_ == i].tolist())

    t_student_cluster_end = datetime.now()

    # logger.info("Student Clustering took | %.3f secs.",
    #             time_diff(t_student_cluster_start, t_student_cluster_end))

    return cluster_ids, cluster_centers, cluster_students


def get_transformed_coordinates(df_input, center_line):
    """ Get transformed coodinates after rotation, x-stretch and y-compress"""

    # Initialize clustering input variables
    df_cluster_input = df_input.copy(deep=True)

    # get transformed center lines and positions based on center line angle from y axis

    theta = get_y_theta(center_line)

    trans_cline = [rotate(center_line[0][0], center_line[0][1], theta),
                   rotate(center_line[1][0], center_line[1][1], theta)]
    trans_cline = [list([trans_cline[0][0], trans_cline[0][1]]), list([trans_cline[1][0], trans_cline[1][1]])]

    df_cluster_input['transformed_loc'] = df_cluster_input.apply(lambda row: rotate(row['loc_x'], row['loc_y'], theta),
                                                                 axis=1)

    # Center the transformed center line, stretch in x direction, and reposition

    df_cluster_input['cx'] = (df_cluster_input['transformed_loc'].apply(lambda x: x[0]))
    df_cluster_input['cy'] = (df_cluster_input['transformed_loc'].apply(lambda x: x[1]))

    # Offset trans cline to img center in x direction

    x_offset = trans_cline[0][0] - (Constants.FULL_RES_X_MAX / 2)
    df_cluster_input['cx'] = df_cluster_input['cx'] - x_offset
    trans_cline[0][0] = trans_cline[0][0] - x_offset
    trans_cline[1][0] = trans_cline[1][0] - x_offset

    # Offset transline center to transline center
    df_cluster_input['cx'] = df_cluster_input['cx'] - trans_cline[0][0]

    # Stretch around transline center
    x_stretch = get_x_stretch(center_line)
    df_cluster_input['cx'] = x_stretch * df_cluster_input['cx']

    # re-offset from transline center
    df_cluster_input['cx'] = df_cluster_input['cx'] + trans_cline[0][0]

    # Offset y to camera frame as much as possible
    min_y, max_y = df_cluster_input['cy'].min(), df_cluster_input['cy'].max()

    if min_y < 0:
        df_cluster_input['cy'] = df_cluster_input['cy'] + abs(min_y)
        trans_cline[0][1] += abs(min_y)
        trans_cline[1][1] += abs(min_y)

    if max_y > Constants.FULL_RES_Y_MAX:
        df_cluster_input['cy'] = df_cluster_input['cy'] - abs(max_y - Constants.FULL_RES_Y_MAX)
        trans_cline[0][1] -= abs(max_y - Constants.FULL_RES_Y_MAX)
        trans_cline[1][1] -= abs(max_y - Constants.FULL_RES_Y_MAX)

    return df_cluster_input


def rotate(x_orig, y_orig, theta):
    """ Return transformed coordinates in 2D plane after theta rotation """
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    x_new = x_orig * c - y_orig * s
    y_new = x_orig * s + y_orig * c
    return int(x_new), int(y_new)


def get_y_theta(cline):
    """Get angle formed by a line with y axis"""
    p1 = cline[0]
    p2 = cline[1]
    theta = np.arctan((p1[0] - p2[0]) / (p1[1] - p2[1]))
    return theta


def get_x_theta(cline):
    """Get angle formed by a line with x axis"""
    p1 = cline[0]
    p2 = cline[1]
    theta = np.arctan((p1[1] - p2[1]) / (p1[0] - p2[0]))
    return theta


def get_x_stretch(cline, max_stretch=10):
    """ Calculated estimated stretch in x direction based on camera position within a threshold"""
    theta = get_y_theta(cline)
    return min(1 + np.tan(theta), max_stretch)


def get_y_compress(cline, max_compress=1 / 10):
    """ Calculated estimated compress in y direction based on camera position within a threshold"""
    theta = get_y_theta(cline)
    return max(1 - np.tan(theta), 1 - max_compress)

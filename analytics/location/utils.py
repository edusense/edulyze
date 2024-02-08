"""
Author: Anonymized
Created: Anonymized

Util functions for location modules
"""

from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import statistics
import math
import numpy as np

# Import project level functions and classes
from configs.constants import Constants, exitStatus

body_pose_indices = {
    'head': [0],
    'chest': [1],
    'shoulder': [2, 5],
    'elbows': [3, 6],
    'hands': [4, 7],
    'hips': [8, 9, 12],
    'knees': [10, 13],
    'feet': [11, 14],
    'face': [15, 16, 17, 18],
    'soles': [22, 19],
    'toes': [23, 20],
    'heels': [24, 21],
}

student_location_config = {
    'SETTLED_MIN_FRACTION': 0.8,
    'SETTLED_MIN_VARIANCE': 5,
    'SETTLED_MAX_VARIANCE_PERCENTILE': 0.999,
    'MIN_SILHOUETTE_SCORE': 0.2,
    'BLOCK_SETTLED_MODE_PERCENTILE': 0.9
}


def get_location_category(x, y, cline_slope, cline_intercept):
    """ Get location of student in terms f left/right side of class"""
    if (y - (cline_slope * x) - cline_intercept) < 0:
        return 'right'
    else:
        return 'left'


def get_line_parameters(line):
    """ get slope and intercept of line in 2D space"""
    if line[0][0] == line[1][0]:
        slope = np.inf
        intercept = np.inf
    else:
        slope = (line[0][1] - line[1][1]) / (line[0][0] - line[1][0])
        intercept = line[1][1] - (slope * line[1][0])

    return slope, intercept


def get_hip_location(body_kps):
    """Get hip location based on body keypoints based on detection confidences"""
    hip_idx = body_pose_indices.get('hips')
    x_vals, y_vals = [], []
    for idx in hip_idx:
        if body_kps[3 * idx + 2] > 0:
            x_vals.append(body_kps[3 * idx])
            y_vals.append(body_kps[3 * idx + 1])
    if len(x_vals) == 0:  # hip location not found, return average across all body kps found
        for idx in range(15):
            if body_kps[3 * idx + 2] > 0:
                x_vals.append(body_kps[3 * idx])
                y_vals.append(body_kps[3 * idx + 1])
    return np.mean(x_vals), np.mean(y_vals)


def get_shoulder_location(body_kps):
    """Get shoulder location based on body keypoints based on detection confidences"""
    shoulder_idx = body_pose_indices.get('shoulder')
    x_vals, y_vals = [], []
    for idx in shoulder_idx:
        if body_kps[3 * idx + 2] > 0:
            x_vals.append(body_kps[3 * idx])
            y_vals.append(body_kps[3 * idx + 1])
    if len(x_vals) == 0:  # location not found
        return np.nan, np.nan
    return np.mean(x_vals), np.mean(y_vals)


def get_head_location(boundingBox):
    """Get head location based on bounding box"""
    if boundingBox is None:
        return np.nan, np.nan
    else:
        return (boundingBox[0][0] + boundingBox[1][0]) / 2, (boundingBox[0][1] + boundingBox[1][1]) / 2


def diff_head_bb(student_df):
    """ Calculate change in head position between bb1 and bb based on center change"""

    if student_df.shape[0] <= 1:
        return np.array([0.])
    student_df['boundingBox'] = student_df['boundingBox'].fillna(method='ffill')

    student_df.loc[student_df['boundingBox'].isnull(), 'boundingBox'] = 0.  # make all other null values equal to zero

    diff_func = lambda x: ((np.absolute(
        np.array(x[1]) - np.array(x[3])).mean()) * Constants.MILLISECS_IN_SEC / np.absolute(x[0] - x[2])) if (
            abs(x[0] - x[2]) > 0) else 0

    diff_input = np.concatenate([student_df.values[:-1, :], student_df.values[1:, :]], axis=1)

    diff_output = np.apply_along_axis(diff_func, 1, diff_input)
    diff_output = np.insert(diff_output, 0, 0)

    return diff_output


def diff_body_kps(student_df):
    """ Calculate change in body position between kps1 and kps2 based on centroid change"""

    if student_df.shape[0] <= 1:
        return np.array([0.])
    diff_func = lambda x: ((np.absolute(
        np.array(x[1]) - np.array(x[3])).mean()) * Constants.MILLISECS_IN_SEC / np.absolute(x[0] - x[2])) if (
            abs(x[0] - x[2]) > 0) else 0
    diff_input = np.concatenate([student_df.values[:-1, :], student_df.values[1:, :]], axis=1)
    diff_output = np.apply_along_axis(diff_func, 1, diff_input)
    diff_output = np.insert(diff_output, 0, 0)
    return diff_output


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


def perc_of_list(ls, val):
    """ calculate fraction of a value over a list"""
    return list(ls).count(val) / len(ls)


def longest_consecutive(ls, val):
    """ calculate longest contegious array of a value over a list"""
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


# determine whether insturctor at location (board, podium, etc.)
#   row: ins. frame row
#   loc_ojb: polygon of object in question
def at_loc(row, loc_objs):
    """ check weather a particular person in particular frame is near given location"""
    is_at_loc = False
    for loc_obj in loc_objs:
        if row["gazeVector"] is not None:
            loc_x = row["gazeVector"][0][0]
            loc_y = row['gazeVector'][0][1]
        elif row['loc'] is not None:
            loc_x = row["loc"][0]
            loc_y = row['loc'][1]
        else:
            return False

        poly = Polygon(loc_obj)
        # get minimum bounding box around polygon
        box = poly.minimum_rotated_rectangle

        # get coordinates of polygon vertices and length of bounding box edges
        x, y = box.exterior.coords.xy
        edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))

        # length = longest side, width = shortest sie
        # length = max(edge_length)
        width = min(edge_length)

        ins_loc = Point(loc_x, loc_y)
        is_at_loc = is_at_loc | (poly.exterior.distance(ins_loc) < width)
    return is_at_loc


def moving(row1, row2, stand_still_radius):
    """  determine if person is moving from one fram to another"""

    if row1["gazeVector"] is not None:
        loc_x_row1 = row1["gazeVector"][0][0]
        loc_y_row1 = row1['gazeVector'][0][1]
    elif row1['loc'] is not None:
        loc_x_row1 = row1["loc"][0]
        loc_y_row1 = row1['loc'][1]
    else:
        return False

    if row2["gazeVector"] is not None:
        loc_x_row2 = row2["gazeVector"][0][0]
        loc_y_row2 = row2['gazeVector'][0][1]
    elif (row2['loc'] is not None) & (row1['gazeVector'] is None):
        loc_x_row2 = row2["loc"][0]
        loc_y_row2 = row2['loc'][1]
    else:
        return False

    # radius within which considered still
    # buffer = 10

    start = Point(loc_x_row1, loc_y_row1)
    end = Point(loc_x_row2, loc_y_row2)

    dist = start.distance(end)

    if dist > 1000000000:
        return None

    return dist > stand_still_radius


# determine side of class instructor is on
def class_side(row, div):
    """determine side of class a person is on"""
    if row["gazeVector"] is not None:
        loc_x = row["gazeVector"][0][0]
        loc_y = row['gazeVector'][0][1]
    elif row['loc'] is not None:
        loc_x = row["loc"][0]
        loc_y = row['loc'][1]
    else:
        return "left"

    return (place_target(div[0], div[1], div[2], div[3], loc_x, loc_y))


def place_target(x1, y1, x2, y2, p_x, p_y):
    """helper function for class side function"""
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


def head_entropy(row1, row2):
    """determine positional entropy of head keypoints"""
    if row1["gazeVector"] is None or row2["gazeVector"] is None:
        return 0

    x = row2['gazeVector'][0][0] - row1['gazeVector'][0][0]
    y = row2['gazeVector'][0][1] - row1['gazeVector'][0][1]

    return (entropy_helper(x, y))


def body_entropy(row):
    """determine entropy of body centroid keypoint"""
    if row["centroidDelta"] is None:
        return 0

    c1, c2 = row["centroidDelta"][0], row["centroidDelta"][1]

    return entropy_helper(c1, c2)


def entropy_helper(p1, p2):
    """Helper function for entropy functions"""
    return math.sqrt(math.pow(p1, 2) + math.pow(p2, 2))

def get_instructor_movement(df_instructor_location, stand_still_distance, stand_still_duration='00:00:10'):
    # Cluster datapoints as stops and transitions
    df_instructor_location['timestamp'] = pd.to_datetime(df_instructor_location['timestamp'],unit='ms')
    df_instructor_location["x"] = df_instructor_location['loc'].apply(lambda x: x[0])
    df_instructor_location["y"] = df_instructor_location['loc'].apply(lambda x: x[1])
    # df_instructor_location["x"]
    df = generate_positioning_clusters(df_instructor_location, stand_still_distance, stand_still_duration)
    print("Generating clusters completed")

    # Tag clusters as stops or transition
    df = tag_clusters(df, stand_still_distance, stand_still_duration)
    print("Clusters tagged")

    # Generate data frame with information about stops and transitions to be further processed to generate metrics
    df = get_stops_and_transitions(df)
    print("Data frame of stops and transitions generated")

    print("Processing stops and transitions COMPLETED")
    return (df)


def generate_positioning_clusters(df, stand_still_distance,stand_still_duration):
    df_dist = pd.DataFrame()

    list_trackingId = df.trackingId.unique()
    # list_session = df.session.unique()

    for x, i in enumerate(list_trackingId):

        ## Loop through unique trackingId and session
        # df = df.copy()
        trackingId2 = i
        trackingId_df = df[df['trackingId'] == trackingId2].sort_values(by=['timestamp'])

        # for y, j in enumerate(list_session):
        # session2 = j
        # session_df = trackingId_df[trackingId_df['session'] == session2]

        # Shift x and y to get the coordinates of the previous positioning datapoint
        trackingId_df['x_shifted'] = trackingId_df['x'].shift(1)
        trackingId_df['y_shifted'] = trackingId_df['y'].shift(1)

        # Remove first row of every session (shifted = nan)
        trackingId_df = trackingId_df.dropna()

        # Calculate distance between two points
        trackingId_df['dist'] = np.sqrt(
            (trackingId_df['x_shifted'] - trackingId_df['x']) ** 2 + (trackingId_df['y_shifted'] - trackingId_df['y']) ** 2)

        # Append Distance column
        df_dist = df_dist.append(trackingId_df)

    # CREATE CLUSTERS OF DATA POINTS (STOPS) according to Distance and Duration parameters

    # Load parameter that is used to create a new cluster if the distance between two consecutive datapoints is
    # higher than the parameter 'distance'
    distance = stand_still_distance

    # sequential numbering of clusters of positioning datapoints (called in this code "group or grouping")
    fix_seq = 1  # fixation sequence, starts at 1 (then adds +1) --> to create group numbering

    data1 = []  # this will be used to save clusters

    ## Loop through unique trackingId and session

    previous_row = None
    for x, i in enumerate(list_trackingId):

        trackingId = i
        trackingId_df = df[df['trackingId'] == trackingId]

        # for y, j in enumerate(list_session):
        count = 0
        #     session = j
        #     session_df = trackingId_df[trackingId_df['session'] == session]

        # print details of each trackingId and seesion while code is running (useful in identifying if code breaks)
        print(trackingId, len(trackingId_df))

        # Convert dataframe to list
        # data = session_df.values.tolist()
        data = trackingId_df

        ## Loop through each row
        # if row (k) > 0; if <= t_hold then calculate distance to base point, and add to time
        #                if > t_hold - then redefine base point and start calculating distance to base point and add time
        # if row (k) = 0; then use the values as is (see elif k == 0)

        for k, row_pair in data.iterrows():
            count = count + 1
            # for k, i in enumerate(data[0:]):
            if count > 1:
                p_time = row_pair["timestamp"]

                # x = column [5]; y = column [6]
                x2 = float(row_pair["x"])  # current row value
                x1 = float(previous_row["x"])  # previous row value
                y2 = float(row_pair["y"])
                y1 = float(previous_row["y"])
                phase = 1
                quantile = 1

                # calculate the distance from previous point
                dist_p = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                # calculate the distance from base point (the first point in the group/cluster)
                dist_b = math.sqrt((x2 - b_x1) ** 2 + (y2 - b_y1) ** 2)
                # caluclate the time delta from base point
                t_diff = p_time - b_time

                if dist_b <= distance:
                    # if distance from current point to centroid is lower than threshold (distance = 1000mm - 1m),
                    # it means that the person/object is covering a short distance - the point is inside the cluster
                    # add to list with current cluster/group number
                    data1.append(
                        [fix_seq, trackingId, phase, quantile, str(p_time), x2, y2, dist_b, dist_p, t_diff])

                elif dist_b > distance:
                    # if distance from current point to centroid is higher than threshold (distance = 1000mm - 1m),
                    # it means that the person/object is in another cluster - then change centroid info
                    # Create new cluster or grouping
                    b_time = row_pair["timestamp"]
                    b_x1 = float(row_pair["x"])
                    b_y1 = float(row_pair["y"])
                    phase = 1
                    quantile = 1

                    fix_seq = fix_seq + 1  # coordinate grouping according to distance threshold
                    data1.append([fix_seq, trackingId, phase, quantile, b_time, b_x1, b_y1, dist_b, dist_p])
                # don't add t_diff as 0; leaving blank will introduce NaT (which will identify the column as timedate type)

                t_diff = p_time - b_time
                previous_row = row_pair

            elif count == 1:
                previous_row = row_pair
                b_time = row_pair["timestamp"]
                b_x1 = float(row_pair["x"])
                b_y1 = float(row_pair["y"])
                phase = 1
                quantile = 1
                data1.append([fix_seq, trackingId, phase, quantile, b_time, b_x1, b_y1, 0, 0])
            # don't add t_diff as 0; leaving it blank will introduce NaT (which will identify the column as timedate type)

    df2 = pd.DataFrame(data1,
                       columns=['group', 'trackingId', 'phase', 'quantile', 'timestamp', 'x', 'y', 'base_dist',
                                'intra_dist', 'time_diff'])
    return (df2)

def tag_clusters(df_dist, stand_still_distance,stand_still_duration):

    # Extract the number of seconds component from column "time_diff"
    df_dist[['time_diff']] = df_dist[['time_diff']].astype(str)
    # Get seconds from the time difference
    df_dist['time_diff2'] = df_dist['time_diff'].str.extract('(..:..:..)', expand=True)
    df_dist = df_dist.fillna('00:00:00')
    df_dist['time_diff2'] = pd.to_timedelta(df_dist['time_diff2'])
    df_dist = df_dist[
        ['group', 'trackingId', 'phase', 'quantile', 'timestamp', 'x', 'y', 'base_dist', 'intra_dist',
         'time_diff2']]

    # create 'max_duration' column  - to use it to further define stops and transitions
    df_dist['max_duration'] = df_dist.groupby(['group'])['time_diff2'].transform(max)
    df_dist['max_duration'] = pd.to_timedelta(df_dist['max_duration'])

    # Add Type (Stop and Transition) column
    # Assign stop and transition labels according to parameter duration

    # get parameter from config file
    duration = stand_still_duration

    # Tag clusters as stops or transitions
    type = []
    stop = pd.Timedelta(duration)

    for row in df_dist['max_duration']:
        if row >= stop:
            type.append('stop')
        else:
            type.append('transition')

    # Add a column to the data frame from the list 'type'
    df_dist['type'] = type

    # The following code fixes the "group" column by setting the same ID for all the consecutive datapoints
    # labelled as transition (including clusters with less datapoints than the parameter 'duration').
    # Fuller explanation: A Transition (sequential) can be made of several 'clusters' (due to distance with base
    # exceeding threshold set (e.g. 1m)), this is because when distance exceeds 1m, the numbering resets for the
    # column "group", therefore starting with a new group or cluster of datapoints. This means that the
    # 'group' column would have several groups which belong to the same transition. Since in actuality
    # these transitions belong to a single transition. Consequently, this step is required so that these
    # transitions are labelled to belong to the same one transition.

    # A new column "block" is added to uniquely identify the stop or transition. A block can contain multiple 'groups'

    df_group = df_dist.copy()
    df_new_group = pd.DataFrame()
    list_trackingId = df_group.trackingId.unique()
    # list_session = df_group.session.unique()

    ## Loop through unique group
    for x, i in enumerate(list_trackingId):

        df = df_group.copy()
        trackingId = i
        trackingId_df = df[df['trackingId'] == trackingId]

        # for y, j in enumerate(list_session):
        #     session = j
        #     session_df = trackingId_df[trackingId_df['session'] == session]

        trackingId_df['movement_block'] = (trackingId_df.type.shift(1) != trackingId_df.type).astype(int).cumsum()

        # SAVE
        df_new_group = df_new_group.append(trackingId_df)

    # Duration between each point (by movement_block)
    # Since sequential transitions are relabelled as belonging to a single group.
    # The duration between each point needs to be recalculated as well (this since, duration resets to
    #	'00:00:00' with the distance calculation performed in function generate_positioning_clusters(df))
    df_dur = df_new_group.copy()

    trackingId = df_dur.trackingId.unique()

    # session = df_dur.session.unique()

    movement_block = df_dur.movement_block.unique()

    df_new = pd.DataFrame()

    list_trackingId = trackingId
    list_movement_block = movement_block

    for x, i in enumerate(list_trackingId):

        df = df_dur.copy()
        trackingId = i
        trackingId_df = df[df['trackingId'] == trackingId]
        # trackingId_df
        #
        # for y, j in enumerate(list_session):
        #
        #     session = j
        #     session_df = trackingId_df[trackingId_df['session'] == session]

        for z, k in enumerate(list_movement_block):
            movement_block = k
            movement_block_df = trackingId_df[trackingId_df['movement_block'] == movement_block]

            movement_block_df['delta'] = (movement_block_df['timestamp'] - movement_block_df['timestamp'].shift()).fillna(
                pd.Timedelta('0 days'))

            # SAVE
            df_new = df_new.append(movement_block_df)

    return (df_new)


def get_stops_and_transitions(df_dist):
    """This function generates a data frame that contains meta data about stops  (one per line) and transitions (all the lines
    to enable further modelling of the trajectory itself)

    Parameters
    ----------
    df_dist : Pandas Data Frame
        This df has to be the Data frame returned by the function: tag_clusters(df)
        It has to contain the following columns:
            type - (string) stop or transition
            movement_block - (int) the unique identifier of the stop or transition
            ------------------plus the original columns of the dataset, for example:
            timestamp (datetime as "%Y-%m-%d_%H:%M:%S")
            session (identifier)
            trackingId (identifier)
            x and y (coordinates)
            phase (int)
            quantile (int) Set to 1 if not interested in using this column

    Returns
    -------
    merge
        returns a data frame with the following additional columns
            x, y - point at the centroids of the stops
            x_stdev, y_stdev - standard deviation of the points within a stop. (for transitions the value is zero)

    """

    list_trackingId = df_dist.trackingId.unique()
    # list_session = df_dist.session.unique()
    list_movement_block = df_dist.movement_block.unique()

    #### 1) Max duration
    # Once the sequential saccades are assign to a movement_block, the max duration for each movement_block is derived
    df_max_duration = pd.DataFrame()

    for x, i in enumerate(list_trackingId):

        df = df_dist.copy()
        trackingId = i
        trackingId_df = df[df['trackingId'] == trackingId]
        trackingId_df

        # for y, j in enumerate(list_session):
        #     session = j
        #     session_df = trackingId_df[trackingId_df['session'] == session]

        trackingId_df['max_duration'] = trackingId_df.groupby('movement_block')['timestamp'].transform(
            lambda x: x.iat[-1] - x.iat[0])
        print(trackingId_df['max_duration'])
        # trackingId_df.groupby('movement_block')['timestamp'].transform(lambda x: x.iat[-1] - x.iat[0]).to_timedelta().total_seconds()

        # SAVE
        df_max_duration = df_max_duration.append(trackingId_df)

    # GET STOPS
    df_dur_fixation = df_max_duration[df_max_duration['type'] == 'stop']  # filter only stops

    # GET TRANSITIONS
    df_dur_saccades = df_max_duration[df_max_duration['type'] == 'transition']  # filter only transitions

    # Extract just the movement_block and max duration, and remove duplicate rows
    df_duration = df_dur_fixation[['trackingId', 'movement_block', 'phase', 'quantile', 'max_duration', 'type']]
    df_duration = df_duration.drop_duplicates()
    df_duration.sort_values(by=['trackingId', 'movement_block'], inplace=True)

    df_duration = df_duration.groupby(['trackingId', 'movement_block', 'max_duration', 'type']).agg(
        {
            'phase': min,
            'quantile': 'first'  # get the first date per group
        }
    )
    df_duration.reset_index(inplace=True)
    df_duration.sort_values(by=['trackingId', 'movement_block'], inplace=True)

    #### 2) Get first and last timestamp value for each movement_block
    # Timestamp - start and end (for entire dataset - fixation and saccades)
    df = df_max_duration.copy()
    df = df.groupby(['trackingId', 'movement_block'])['timestamp'].agg(['first', 'last'])
    df.index = df.index.set_names(['trackingId', 'movement_block'])
    df.reset_index(inplace=True)
    df.rename(columns={'first': 'timestamp'}, inplace=True)

    #### 3) Centroid (x,y)
    # Calculate centroid (mean of x and y coordinate by movement_blocks) - only for fixation
    test = df_dur_fixation.copy()

    test2_centroid = test.groupby(['trackingId', 'movement_block'])['x', 'y'].agg(['mean'])
    test2_centroid.columns = ['x', 'y']  # relabel column (x,y)
    test2_centroid.index = test2_centroid.index.set_names(['trackingId', 'movement_block'])
    test2_centroid.reset_index(inplace=True)
    test2_centroid.head()

    #### 4) Stdev (x,y) - only for fixation
    test2_stdev = test.groupby(['trackingId', 'movement_block'])['x', 'y'].agg(['std'])
    test2_stdev.columns = ['x_stdev', 'y_stdev']  # relabel column (x,y)
    test2_stdev.index = test2_stdev.index.set_names(['trackingId', 'movement_block'])
    test2_stdev.reset_index(inplace=True)
    test2_stdev.head()

    #### 5) Tracker and Session info
    df_trackingId = df_dist.copy()
    df_trackingId = df_trackingId[['movement_block', 'trackingId']]
    df_trackingId = df_trackingId.drop_duplicates()
    df_trackingId.head()

    ### 7) Merge
    # Create dataframe with additional information for the movement_blocks (max duration, centroid, stdev)

    # remove phase and quartile; add timestamp (start time and end time)  <--------------------------------avoid doing this!
    merge = pd.merge(pd.merge(test2_centroid, test2_stdev, on=['trackingId', 'movement_block']), df_duration,
                     on=['trackingId', 'movement_block'])

    # Merge with start and end timestamp
    merge = pd.merge(merge, df, on=['trackingId', 'movement_block'])

    # merge with transitions
    merge = merge.append(df_dur_saccades)
    # order by movement_block
    merge = merge.sort_values(['trackingId', 'movement_block', 'timestamp'])
    merge = merge.fillna(pd.Timedelta(seconds=0))
    merge['max_duration_sec'] = merge['max_duration'].dt.total_seconds()

    return (merge)


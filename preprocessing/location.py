"""
This files consists of common preprocessing for location data in edulyze. it includes
1. syncing ids across frames( for video based data)
2. Imputing missing data across frames.

Author: Anonymized
Created at: Anonymized
"""

# basic libraries
import numpy as np
import pandas as pd

# custom libraries
from configs.constants import Constants


def preprocess_location_data(df_raw_location_data, raw_meta_data_dict, logger):
    """
    This is main wrapper function for preprocessing location formatted in a specific format
    :param df_raw_location_data:
    :param logger:
    :return:
    df_processed_location_data: Final location information after all processing
    location_preprocessing_metrics: location preprocessing metrics
    """
    df_processed_location_data = None
    location_preprocessing_metrics = dict()

    df_processed_location_data = df_raw_location_data.copy(deep=True)
    session_start_timestamp =raw_meta_data_dict['session_start_timestamp']
    df_processed_location_data['block_id'] = ((df_processed_location_data['timestamp'] / Constants.MILLISECS_IN_SEC)
                                                 - session_start_timestamp) // Constants.BLOCK_SIZE
    df_processed_location_data['block_id'] = df_processed_location_data['block_id'].astype(int)



    df_student_location_data = df_raw_location_data[df_raw_location_data.channel == 'student']
    df_instructor_location_data = df_raw_location_data[df_raw_location_data.channel == 'instructor']

    ################### Preprocessing Student Data ###################
    # ---------Part A: Making ids sync better with frame shift and location proxemics---------

    if df_student_location_data.shape[0]==0:
        logger.error("No student location data available, skipping id syncing...")
        return df_processed_location_data, location_preprocessing_metrics


    df_ids_raw = df_student_location_data[['timestamp', 'frameNumber', 'trackingId', 'loc']]
    df_ids_raw['loc'] = df_ids_raw['loc'].apply(lambda x: np.sqrt(x[0]**2 + x[1]**2 + x[2]**2))

    # Count null ids and add random ids to null ids parts
    max_id = df_ids_raw.trackingId.max()
    count_null_ids = df_ids_raw.trackingId.isnull().sum()
    if count_null_ids > 0.:
        df_ids_raw.loc[df_ids_raw.trackingId.isnull(), 'trackingId'] = max_id + 1 + np.arange(count_null_ids)

    # ---------Part B: Sync Ids based on frameStart, frameStop, and location proxemics---------
    id_ts_matrix = pd.pivot_table(df_ids_raw.sort_values(by=['timestamp', 'trackingId']), index='timestamp', columns='trackingId',
                                  values='loc', aggfunc='mean')
    df_id_info = df_ids_raw.groupby('trackingId', as_index=False).agg({
        'timestamp': ['min', 'max'],
        'loc': ['min', 'max']
    })
    df_id_info.columns = ['trackingId', 'min_ts', 'max_ts', 'first_loc', 'last_loc']
    df_id_info = df_id_info.sort_values(by=['min_ts', 'trackingId'])

    id_map = {}
    ts_threshold = 10 * Constants.MILLISECS_IN_SEC  # in millisecs
    loc_threshold = 0.1 * df_ids_raw['loc'].max()
    for idx, row in df_id_info.iterrows():
        row_id = row['trackingId']
        eligible_pairs = df_id_info[
            (df_id_info.min_ts > row['max_ts']) &
            (df_id_info.min_ts <= row['max_ts'] + ts_threshold) &
            (np.absolute(df_id_info.first_loc - row['last_loc']) <= loc_threshold)]
        if eligible_pairs.shape[0] > 0:
            id_to_map = eligible_pairs.sort_values(by=['min_ts']).trackingId.values[0]
            id_map[id_to_map] = row_id

    id_to_map_list = list(id_map.keys())
    new_id_map = dict()
    for id_to_map in id_map.keys():
        final_map_id = id_map[id_to_map]
        while final_map_id in id_to_map_list:
            final_map_id = id_map[final_map_id]
        new_id_map[id_to_map] = final_map_id

    # make changes in raw id info
    for id_to_map in new_id_map.keys():
        id_mapped_on = new_id_map[id_to_map]
        df_ids_raw.loc[df_ids_raw.trackingId == id_to_map, 'trackingId'] = id_mapped_on

    id_ts_matrix = pd.pivot_table(df_ids_raw.sort_values(by=['timestamp', 'trackingId']), index='timestamp', columns='trackingId',
                                  values='loc', aggfunc='mean')
    df_id_info = df_ids_raw.groupby('trackingId', as_index=False).agg({
        'timestamp': ['min', 'max'],
        'loc': ['min', 'max']
    })
    df_id_info.columns = ['trackingId', 'min_ts', 'max_ts', 'first_loc', 'last_loc']
    df_id_info = df_id_info.sort_values(by=['min_ts', 'trackingId'])

    # replace these ids with usual ids
    id_original = df_student_location_data['trackingId']
    id_mapped = df_ids_raw['trackingId']
    id_map_records = pd.concat([id_original,id_mapped],axis=1,ignore_index=True)

    df_student_location_data['trackingId'] = df_ids_raw['trackingId']

    # remove ids with only less than 5 ptile of occurence
    id_counts = df_student_location_data['trackingId'].value_counts()
    min_count_threshold = np.percentile(id_counts, 5)
    ids_to_remove = id_counts[id_counts<=min_count_threshold].index.values
    df_student_location_data = df_student_location_data[~(df_student_location_data.trackingId.isin(ids_to_remove))]
    # --------- Part C: Impute Missing location data at id level ---------
    # student_id_dfs = []
    # # timestamp_vals
    # student_nan_fraction = []
    # for id in df_student_location_data.trackingId.unique():
    #     df_student_id = df_student_location_data[df_student_location_data.trackingId==id].sort_values(by='timestamp')
    #     id_min_ts, id_max_ts = df_student_id.timestamp.min(), df_student_id.timestamp.max()
    #
    #
    #     student_nan_fraction.append(df_student_id.bodykps.isnull())
    #     df_student_id = df_student_id.fillna('ffill')


    ## Add location details based on loc value of not present

    # get derivative features
    # if 'gazeVector' not in df_processed_location_data:


    # df_raw_location['gazeVector'] = None
    # df_raw_location['orientation'] = None
    # df_raw_location['boundingBox'] = None
    # df_raw_location['centroidDelta'] = None
    #
    # # set unavailable information to None
    # df_raw_location['body_kps'] = None
    # df_raw_location['armPose'] = None
    # df_raw_location['sitStand'] = None

    return df_processed_location_data, location_preprocessing_metrics

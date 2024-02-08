"""
This is driver to interface with Moodoo system.
general information: Moodoo collects gaze, location and audio data at 3-15FPS using two cameras, and stores raw information
in mongodb, which can be extracted
"""

# basic libraries
import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime
from copy import deepcopy
import pickle
import configparser

# custom libraries
from drivers.DriverInterface import DriverInterface, MethodNotImplementedException
from utils.time_utils import time_diff
from drivers.moodoo.utils.cache_session_data import cache_session_data, exitStatus
from analytics.location.utils import get_hip_location
from configs.constants import Constants
from pii_information.location_config import get_location_config


class MoodooDriver(DriverInterface):
    """
    This is driver implementation to fetch raw information from moodoo mongo database
    """

    def __init__(self, run_config: dict, logger: logging.LoggerAdapter) -> None:
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and makes it ready for use.
        The __init__ function can take arguments, but self must be its first argument.

        Args:
            self: Represent the instance of the class
            run_config: dict: Pass the configuration parameters to the driver
            logger: logging.LoggerAdapter: Log the information about the driver

        Returns:
            None

        Doc Author:
            Trelent
        """
        DriverInterface.__init__(self, run_config, logger)  # initialize basic driver

        # Fetch all data from cache files and store raw information
        t_data_fetch_start = datetime.now()
        if run_config.get("cache_mode") & (not run_config.get("cache_update")):

            # Get information about cache path from run config

            session_id = run_config.get("session_id")
            server_name = 'moodoo_files'
            cache_dir = run_config.get("cache_dir")

            # try to fetch data from cache

            try:
                with open(f"{cache_dir}/{server_name}/{session_id}.pb", "rb") as f:
                    session_cached_input = pickle.load(f)

                # Update this run config to make sure we have all config related changes
                session_cached_input.update(run_config)
                t_data_fetch_end = datetime.now()

                self.session_input = session_cached_input
                logger.info("Initialization of input object from cache took | %.3f secs.",
                            time_diff(t_data_fetch_start, t_data_fetch_end))

                return None
            except:
                logger.warning("Cache mode is true. Unable to get file from cache")
        else:
            logger.info("Reading from cache disabled. either because cache mode is false or cache update is true.")
            logger.info("Not reading from cache.")

        # fetch raw frame data, tracker data, phase data and room config

        session_cached_input = dict()
        session_cached_input.update(run_config)
        session_id = run_config['session_id']
        phase_id = run_config['phase_id']

        # Get raw position data for session

        df_position_data = pd.read_csv(f'{run_config["session_data_file"]}')
        df_position_data = df_position_data[(df_position_data.session==session_id) & (df_position_data.phase==phase_id)]
        df_position_data['timestamp'] = pd.to_datetime(df_position_data['timestamp'],format='%d/%m/%Y %H:%M:%S').values.astype(np.int64) // 10**6
        df_position_data['channel']='instructor'
        session_cached_input['instructor_position_df'] = df_position_data

        # get tracker data for session

        df_moving_tracker_data = pd.read_csv(f'{run_config["moving_trackers_config_file"]}')
        df_moving_tracker_data = df_moving_tracker_data[df_moving_tracker_data.session==session_id]
        df_moving_tracker_data['channel'] = 'instructor'

        df_fixed_tracker_data = pd.read_csv(f'{run_config["fixed_trackers_config_file"]}')
        df_fixed_tracker_data = df_fixed_tracker_data[df_fixed_tracker_data.session==session_id]
        df_student_tracker_data = df_fixed_tracker_data[df_fixed_tracker_data.obj_type=='student']
        df_student_tracker_data['channel'] = 'student'
        zone_tracker_data = df_fixed_tracker_data[df_fixed_tracker_data.obj_type=='zone'].set_index('tag').to_dict(orient='index')

        session_cached_input['student_position_df'] = df_student_tracker_data
        session_cached_input['objects_position_data'] = zone_tracker_data

        # fetch phase information for session

        phase_info_df = pd.read_csv(f'{run_config["session_phase_file"]}')
        phase_info_df['start'] = pd.to_datetime(phase_info_df['start'],
                                                       format='%d/%m/%Y %H:%M:%S').values.astype(np.int64) // 10**6
        phase_info_df['end'] = pd.to_datetime(phase_info_df['end'],
                                                format='%d/%m/%Y %H:%M:%S').values.astype(np.int64) // 10**6

        session_phase_info = phase_info_df[(phase_info_df.session==session_id) & (phase_info_df.phase==phase_id)].iloc[0].to_dict()
        session_cached_input['session_phase_info'] = session_phase_info

        #fetch config information for classroom
        config = configparser.ConfigParser()
        config.read(f'{run_config["classroom_config_file"]}')
        config_dict = {s: dict(config.items(s)) for s in config.sections()}['parameters']
        session_cached_input['classroom_config'] = config_dict

        t_data_fetch_end = datetime.now()

        self.session_input = session_cached_input
        logger.info("Initialization of moodoo driver took | %.3f secs.",
                    time_diff(t_data_fetch_start, t_data_fetch_end))

        if self.session_input.get("cache_mode"):
            cache_status = cache_session_data(self.session_input, logger)
            if cache_status == exitStatus.SUCCESS:
                logger.info("Cached Session Data Successfully")
            else:
                logger.error("Caching session data failed with status | %s", cache_status.name)

        return None

    def getLocationInput(self) -> pd.DataFrame:
        """
    The getLocationInput function extracts location data for a single session based on the given run config.
    Location data consist of timestamp, Ids, type('instructor' or 'student'), location coordinates(x,y,z),
    movement info(accel_x, accel_y, accel_z), and body keypoints (if available),
    and inference info (if available)

    Args:
        self: Represent the instance of the class

    Returns:
        A pandas dataframe with location information

    Doc Author:
        Trelent
    """
        logger = self.logger
        run_config = self.run_config
        t_location_input_start = datetime.now()

        # get instructor data for location
        df_instructor_location = self.session_input['instructor_position_df']
        df_instructor_location['trackingId'] = df_instructor_location['tracker']
        df_instructor_location = df_instructor_location[['session','trackingId','channel','timestamp','x','y','z']]

        # filter data for main instructor only
        # not_allowed_instructors = {'Tutor 1':'26656', 'Teacher2D':'Teacher2D'}
        not_allowed_instructors = ['26656','Teacher2D']
        df_timestamps = df_instructor_location[['session','timestamp']].drop_duplicates().sort_values(by='timestamp')
        df_timestamps['frameNumber'] = np.arange(df_timestamps.shape[0])
        df_instructor_location = pd.merge(df_instructor_location, df_timestamps,on=['session','timestamp'],how='left')
        df_instructor_location.trackingId = df_instructor_location.trackingId.astype(str)
        df_instructor_location = df_instructor_location[~df_instructor_location.trackingId.isin(not_allowed_instructors)]
        assert (len(df_instructor_location.trackingId.unique())==1)
        # get student data for location
        # df_student_location = self.session_input['student_position_df']
        # df_student_location['trackingId'] = 'S' + df_student_location['tag'].astype(str)
        # df_student_location['z'] = 0.
        # df_student_location =df_student_location[['session','channel','trackingId']]
        # df_student_location = pd.merge(df_timestamps,df_student_location,on='session',how='left')

        # df_raw_location = pd.concat([df_instructor_location,df_student_location])
        df_raw_location = df_instructor_location
        df_raw_location['x'] = df_raw_location['x']/10
        df_raw_location['y'] = df_raw_location['y']/10
        df_raw_location['z'] = df_raw_location['z']/10
        df_raw_location['loc'] = df_raw_location.apply(
            lambda row: tuple((row['x'], row['y'], row['z'])), axis=1)
        instructor_dfs = []
        for instructor_id in df_raw_location.trackingId.unique():
            ins_df = df_instructor_location[df_instructor_location.trackingId==instructor_id]
            ins_df['x_diff'], ins_df['y_diff'] = ins_df['x'].diff().fillna(0.).values, ins_df['y'].diff().fillna(0.).values
            ins_df['centroidDelta'] = ins_df.apply(lambda row: tuple((row['x_diff'], row['y_diff'])), axis=1)
            instructor_dfs.append(ins_df)
        df_raw_location = pd.concat(instructor_dfs)
        df_raw_location = df_raw_location.sort_values(by=['frameNumber','timestamp','trackingId'])
        df_raw_location['gazeVector'] = None
        t_location_input_end = datetime.now()

        logger.info("Fetching location data from moodoo driver took | %.3f secs.",
                    time_diff(t_location_input_start, t_location_input_end))

        # stand_still_distance = 100
        # df_instructor_movement = get_instructor_movement(df_raw_location, stand_still_distance)

        return df_raw_location
    #
    # def getGazeInput(self) -> pd.DataFrame:
    #     """
    #     The getGazeInput function extracts Gaze data for a single session based on given run config.
    #     Gaze data consist of timestamp, Ids, type('instructor' or 'student'), gaze values(roll,pitch,yaw),
    #     and head bounding boxes (if available)
    #
    #     Args:
    #         self: Represent the instance of the class
    #
    #     Returns:
    #         A dataframe with the following columns:
    #
    #     Doc Author:
    #         Trelent
    #     """
    #     logger = self.logger
    #     run_config = self.run_config
    #     t_gaze_input_start = datetime.now()
    #
    #     # get instructor data for location
    #     df_instructor_gaze = self.session_input['instructor_position_df']
    #     # df_instructor_gaze['trackingId'] = df_instructor_gaze['tracker']
    #     # df_instructor_gaze = df_instructor_gaze[['session','trackingId','channel','timestamp','roll','pitch','yaw']]
    #     #
    #     # df_timestamps = df_instructor_gaze[['session','timestamp']].drop_duplicates().sort_values(by='timestamp')
    #     # df_timestamps['frameNumber'] = np.arange(df_timestamps.shape[0])
    #     # df_instructor_gaze = pd.merge(df_instructor_gaze, df_timestamps,on=['session','timestamp'],how='left')
    #
    #     df_instructor_gaze=None
    #     t_gaze_input_end = datetime.now()
    #
    #     logger.info("Fetching gaze data from moodoo driver took | %.3f secs.",
    #                 time_diff(t_gaze_input_start, t_gaze_input_end))
    #
    #     return df_instructor_gaze
    #
    # def getAudioInput(self) -> pd.DataFrame:
    #     """
    # The getAudioInput function extracts audio data for a single session based on the given run config.
    # Audio data consist of featurized audio in frequency buckets, and MFCC features (if available).
    # The function returns a pandas DataFrame with audio information.
    #
    # Args:
    #     self: Represent the instance of the class
    #
    # Returns:
    #     A dataframe with the following columns:
    #
    # Doc Author:
    #     Trelent
    # """
    #     logger = self.logger
    #     run_config = self.run_config
    #     t_audio_input_start = datetime.now()
    #     df_raw_audio=None
    #     t_audio_input_end = datetime.now()
    #
    #     logger.info("Fetching audio data from moodoo driver took | %.3f secs.",
    #                 time_diff(t_audio_input_start, t_audio_input_end))
    #
    #     return df_raw_audio

    def getMetaInput(self) -> dict:
        """
    The getMetaInput function extracts meta information (required by some modules) for given session, including classroom object positioning,
    camera configs (if data is not captured from 3D sensing but by a camera in classrooms), etc.


    Args:
        self: Represent the instance of the class

    Returns:
        A dict with meta information

    Doc Author:
        Trelent
    """
        logger = self.logger
        location_config_file = self.session_input.get("location_config_file", None)
        session_keyword = self.session_input['session_keyword']
        classroom = "LAB1"

        podium_tags = [kr for kr in self.session_input['classroom_config'] if (self.session_input['classroom_config'][kr]=='Podium')]
        podium_width = int(self.session_input['classroom_config']['podiumwidth'])
        podium_length = int(self.session_input['classroom_config']['podiumwidth'])
        podium_positions =[]
        for tag in podium_tags:
            podium_loc_x,podium_loc_y  = self.session_input['objects_position_data'][tag.upper()]['x'] / 10, self.session_input['objects_position_data'][tag.upper()]['y'] / 10
            podium_positions.append([
                [podium_loc_x-podium_length/2, podium_loc_y-podium_width/2],
                [podium_loc_x+podium_length/2, podium_loc_y-podium_width/2],
                [podium_loc_x+podium_length/2, podium_loc_y+podium_width/2],
                [podium_loc_x-podium_length/2, podium_loc_y+podium_width/2],
            ])

        board_tags = [kr for kr in self.session_input['classroom_config'] if (self.session_input['classroom_config'][kr]=='Board')]
        board_width = int(self.session_input['classroom_config']['boardwidth'])
        board_length = int(self.session_input['classroom_config']['boardwidth'])
        board_positions =[]
        for tag in board_tags:
            board_loc_x,board_loc_y  = self.session_input['objects_position_data'][tag.upper()]['x'] / 10, self.session_input['objects_position_data'][tag.upper()]['y'] / 10
            board_positions.append([
                [board_loc_x-board_length/2, board_loc_y-board_width/2],
                [board_loc_x+board_length/2, board_loc_y-board_width/2],
                [board_loc_x+board_length/2, board_loc_y+board_width/2],
                [board_loc_x-board_length/2, board_loc_y+board_width/2],
            ])


        room_x = int(self.session_input['classroom_config']['room_x']) / 10
        room_y = int(self.session_input['classroom_config']['room_y']) / 10

        common_center_line = [[room_x/2,0], [room_x/2, room_y]]

        class_location_config = {
            'student':{
                'center_line':common_center_line,
            },
            'instructor':{
                'center_line':common_center_line,
                'boards':board_positions,
                'podiums':podium_positions
            }

        }

        meta_input = {
            'classroom':classroom,
            'location_config':class_location_config,
            'stand_still_radius':100,
            'session_start_timestamp': self.session_input['session_phase_info']['start'] //10**3
        }

        return meta_input

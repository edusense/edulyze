"""
This is driver to interface with Edusense system.
general information: Edusense collects gaze, location and audio data at 3-15FPS using two cameras, and stores raw information
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

# custom libraries
from drivers.DriverInterface import DriverInterface, MethodNotImplementedException
from utils.time_utils import time_diff
from drivers.edusense.utils.fetch_input import fetch_audio_data, fetch_video_data
from drivers.edusense.utils.cache_session_data import cache_session_data, exitStatus
from analytics.location.utils import get_hip_location
from configs.constants import Constants
from pii_information.location_config import get_location_config


class EdusenseDriver(DriverInterface):
    """
    This is driver implementation to fetch raw information from edusense mongo database
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

        # Fetch all data from mongodb and store raw information
        t_data_fetch_start = datetime.now()
        if run_config.get("cache_mode") & (not run_config.get("cache_update")):

            # Get information about cache path from run config

            session_id = run_config.get("session_id")
            server_name = run_config.get("server_name")
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

        # fetch raw frame data for audio and video

        session_cached_input = dict()
        session_cached_input.update(run_config)

        # Get raw audio data for session

        raw_audio_data = fetch_audio_data(run_config, logger)
        session_cached_input['audio_data'] = raw_audio_data

        # Fetch raw video data for session

        raw_video_data = fetch_video_data(run_config, logger)
        session_cached_input['video_data'] = raw_video_data

        t_data_fetch_end = datetime.now()

        self.session_input = session_cached_input
        logger.info("Initialization of edusense driver took | %.3f secs.",
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

        #  create a list to store location data
        location_records = []

        # add student data for location

        studentVideoFrames = self.session_input['video_data']['student']['sessions'][0]['videoFrames']

        for studentVideoFrame in studentVideoFrames:
            final_video_frame = {
                'channel': 'student',
                'timestamp': int((studentVideoFrame['timestamp']['unixSeconds'] * Constants.NANOSECS_IN_SEC +
                                  studentVideoFrame['timestamp']['unixNanoseconds']) // 1e6),
                # 'asctime':studentVideoFrame['timestamp']['RFC3339'], # not needed for now
                'frameNumber': studentVideoFrame['frameNumber']
            }
            for personData in studentVideoFrame['people']:
                person_frame = deepcopy(final_video_frame)
                body_kps = personData.get('body', None)
                person_frame['trackingId'] = personData.get('inference', None).get('trackingId', None)

                head_info = personData.get('inference', {}).get('head', {})
                person_frame['gazeVector'] = head_info.get('gazeVector',None)

                face_info = personData.get('inference', {}).get('face', {})
                person_frame['boundingBox'] = face_info.get('boundingBox', None)

                posture_info = personData.get('inference', {}).get('posture', {})
                person_frame['centroidDelta'] = posture_info.get('centroidDelta', None)

                person_frame['body_kps'] = body_kps
                person_frame['body_kps'] = personData.get('body', None)
                person_frame['armPose'] = personData['inference'].get('posture', None).get('armPose', None)
                person_frame['sitStand'] = personData['inference'].get('posture', None).get('sitStand', None)
                person_frame['orientation'] = personData['inference'].get('face', None).get('orientation', None)
                loc_x, loc_y = get_hip_location(body_kps)
                person_frame['loc'] = tuple([loc_x, loc_y, 0.])  # no z direction available
                location_records.append(person_frame)

        # add instructor data for location

        instructorVideoFrames = self.session_input['video_data']['instructor']['sessions'][0]['videoFrames']

        for instructorVideoFrame in instructorVideoFrames:
            final_video_frame = {
                'channel': 'instructor',
                'timestamp': int((instructorVideoFrame['timestamp']['unixSeconds'] * Constants.NANOSECS_IN_SEC +
                                  instructorVideoFrame['timestamp']['unixNanoseconds']) // 1e6),
                # 'asctime':instructorVideoFrame['timestamp']['RFC3339'], # not needed for now
                'frameNumber': instructorVideoFrame['frameNumber']
            }
            for personData in instructorVideoFrame['people']:
                person_frame = deepcopy(final_video_frame)
                body_kps = personData.get('body', None)
                person_frame['trackingId'] = 1

                head_info = personData.get('inference', {}).get('head', {})
                person_frame['gazeVector'] = head_info['gazeVector']

                face_info = personData.get('inference', {}).get('face', {})
                person_frame['boundingBox'] = face_info.get('boundingBox', None)

                posture_info = personData.get('inference', {}).get('posture', {})
                person_frame['centroidDelta'] = posture_info.get('centroidDelta', None)

                person_frame['body_kps'] = body_kps
                person_frame['body_kps'] = personData.get('body', None)
                person_frame['armPose'] = personData['inference'].get('posture', None).get('armPose', None)
                person_frame['sitStand'] = personData['inference'].get('posture', None).get('sitStand', None)
                person_frame['orientation'] = personData['inference'].get('face', None).get('orientation', None)
                loc_x, loc_y = get_hip_location(body_kps)
                person_frame['loc'] = tuple([loc_x, loc_y, 0.])  # no z direction available
                location_records.append(person_frame)

        df_raw_location = pd.DataFrame.from_records(location_records)
        t_location_input_end = datetime.now()

        logger.info("Fetching location data from edusense driver took | %.3f secs.",
                    time_diff(t_location_input_start, t_location_input_end))

        return df_raw_location

    def getGazeInput(self) -> pd.DataFrame:
        """
        The getGazeInput function extracts Gaze data for a single session based on given run config.
        Gaze data consist of timestamp, Ids, type('instructor' or 'student'), gaze values(roll,pitch,yaw),
        and head bounding boxes (if available)

        Args:
            self: Represent the instance of the class

        Returns:
            A dataframe with the following columns:

        Doc Author:
            Trelent
        """
        logger = self.logger
        run_config = self.run_config
        t_gaze_input_start = datetime.now()

        #  create a list to store gaze data
        gaze_records = []

        # add student data for gaze

        studentVideoFrames = self.session_input['video_data']['student']['sessions'][0]['videoFrames']

        for studentVideoFrame in studentVideoFrames:
            final_video_frame = {
                'channel': 'student',
                'timestamp': int((studentVideoFrame['timestamp']['unixSeconds'] * Constants.NANOSECS_IN_SEC +
                                  studentVideoFrame['timestamp']['unixNanoseconds']) // 1e6),
                # 'asctime':studentVideoFrame['timestamp']['RFC3339'], # not needed for now
                'frameNumber': studentVideoFrame['frameNumber']
            }
            for personData in studentVideoFrame['people']:
                person_frame = deepcopy(final_video_frame)
                person_frame['trackingId'] = personData.get('inference', None).get('trackingId', None)

                head_info = personData.get('inference', {}).get('head', {})
                face_info = personData.get('inference', {}).get('face', {})

                person_frame['roll'] = head_info['roll']
                person_frame['pitch'] = head_info['pitch']
                person_frame['yaw'] = head_info['yaw']
                person_frame['gazeVector'] = head_info['gazeVector']
                person_frame['head_bb'] = face_info.get('boundingBox', None)
                person_frame['orientation'] = personData['inference'].get('face', None).get('orientation', None)
                person_frame['armPose'] = personData['inference'].get('posture', None).get('armPose', None)
                person_frame['sitStand'] = personData['inference'].get('posture', None).get('sitStand', None)
                gaze_records.append(person_frame)

        # add instructor data for gaze

        instructorVideoFrames = self.session_input['video_data']['instructor']['sessions'][0]['videoFrames']

        for instructorVideoFrame in instructorVideoFrames:
            final_video_frame = {
                'channel': 'instructor',
                'timestamp': int((instructorVideoFrame['timestamp']['unixSeconds'] * Constants.NANOSECS_IN_SEC +
                                  instructorVideoFrame['timestamp']['unixNanoseconds']) // 1e6),
                # 'asctime':instructorVideoFrame['timestamp']['RFC3339'], # not needed for now
                'frameNumber': instructorVideoFrame['frameNumber']
            }
            for personData in instructorVideoFrame['people']:
                person_frame = deepcopy(final_video_frame)
                person_frame['trackingId'] = personData.get('inference', None).get('trackingId', None)

                head_info = personData.get('inference', {}).get('head', {})
                face_info = personData.get('inference', {}).get('face', {})

                person_frame['roll'] = head_info['roll']
                person_frame['pitch'] = head_info['pitch']
                person_frame['yaw'] = head_info['yaw']
                person_frame['gazeVector'] = head_info['gazeVector']
                person_frame['head_bb'] = face_info.get('boundingBox', None)
                person_frame['orientation'] = personData['inference'].get('face', None).get('orientation', None)
                person_frame['armPose'] = personData['inference'].get('posture', None).get('armPose', None)
                person_frame['sitStand'] = personData['inference'].get('posture', None).get('sitStand', None)
                gaze_records.append(person_frame)

        df_raw_gaze = pd.DataFrame.from_records(gaze_records)
        t_gaze_input_end = datetime.now()

        logger.info("Fetching gaze data from edusense driver took | %.3f secs.",
                    time_diff(t_gaze_input_start, t_gaze_input_end))

        return df_raw_gaze

    def getAudioInput(self) -> pd.DataFrame:
        """
    The getAudioInput function extracts audio data for a single session based on the given run config.
    Audio data consist of featurized audio in frequency buckets, and MFCC features (if available).
    The function returns a pandas DataFrame with audio information.

    Args:
        self: Represent the instance of the class

    Returns:
        A dataframe with the following columns:

    Doc Author:
        Trelent
    """
        logger = self.logger
        run_config = self.run_config
        t_audio_input_start = datetime.now()

        #  create a list to store audio data
        audio_records = []

        # add student data for audio

        studentAudioFrames = self.session_input['audio_data']['student']['sessions'][0]['audioFrames']

        for studentAudioFrame in studentAudioFrames:
            final_audio_frame = {
                'channel': 'student',
                'id':0.,
                'timestamp': int((studentAudioFrame['timestamp']['unixSeconds'] * Constants.NANOSECS_IN_SEC +
                                  studentAudioFrame['timestamp']['unixNanoseconds']) // 1e6),
                # 'asctime':studentVideoFrame['timestamp']['RFC3339'], # not needed for now
                'frameNumber': studentAudioFrame['frameNumber'],
                'amplitude': studentAudioFrame['audio']['amplitude'],
                'melFrequency': studentAudioFrame['audio']['melFrequency'],
                'mfccFeatures': studentAudioFrame['audio'].get('mfccFeatures', None),
                'polyFeatures': studentAudioFrame['audio'].get('polyFeatures', None),
            }
            audio_records.append(final_audio_frame)

        # add instructor data for audio

        instructorAudioFrames = self.session_input['audio_data']['instructor']['sessions'][0]['audioFrames']

        for instructorAudioFrame in instructorAudioFrames:
            final_audio_frame = {
                'channel': 'instructor',
                'id': 0.,
                'timestamp': int((instructorAudioFrame['timestamp']['unixSeconds'] * Constants.NANOSECS_IN_SEC +
                                  instructorAudioFrame['timestamp']['unixNanoseconds']) // 1e6),
                # 'asctime':instructorVideoFrame['timestamp']['RFC3339'], # not needed for now
                'frameNumber': instructorAudioFrame['frameNumber'],
                'amplitude': instructorAudioFrame['audio']['amplitude'],
                'melFrequency': instructorAudioFrame['audio']['melFrequency'],
                'mfccFeatures': instructorAudioFrame['audio'].get('mfccFeatures', None),
                'polyFeatures': instructorAudioFrame['audio'].get('polyFeatures', None),
            }
            audio_records.append(final_audio_frame)

        df_raw_audio = pd.DataFrame.from_records(audio_records)
        t_audio_input_end = datetime.now()

        logger.info("Fetching audio data from edusense driver took | %.3f secs.",
                    time_diff(t_audio_input_start, t_audio_input_end))

        return df_raw_audio

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
        classroom = "_".join(session_keyword.split("_")[-3:-1])

        location_config = get_location_config(classroom, logger, location_config_file)

        meta_input = {
            'classroom':classroom,
            'location_config':location_config,
            'stand_still_radius': 100,
            'session_start_timestamp':int(
            self.session_input['video_data']['instructor']['sessions'][0]['videoFrames'][0]['timestamp']['unixSeconds'])
        }

        return meta_input

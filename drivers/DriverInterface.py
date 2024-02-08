"""
This file contains common interface for all kinds of sensing systems we can connect to edulyze.
Author: Prasoon Patidar
Created at: 26th Sept, 2022
"""
import pandas as pd


class DriverInterface:
    """
    This is interface for drivers to connect with low level sensing systems
    """

    def __init__(self, run_config: object, logger: object) -> None:
        """
        Initializes driver for fetching raw data
        :param run_config:
        :param logger: logging object for drivers
        :return: pd.Dataframe with location information
        """
        self.run_config = run_config
        self.logger = logger

    def getLocationInput(self) -> pd.DataFrame:
        """
        Extracts Location data for a single session based on given run config.
        Location data consist of timestamp, Ids, type('instructor' or 'student'), location coordinates(x,y,z),
        movement info(accel_x, accel_y, accel_z), and body keypoints (if available)

        :return: pd.Dataframe with location information
        """
        raise MethodNotImplementedException("MethodNotImplementedException")

    def getGazeInput(self) -> pd.DataFrame:
        """
        Extracts Gaze data for a single session based on given run config.
        Gaze data consist of timestamp, Ids, type('instructor' or 'student'), gaze values(roll,pitch,yaw),
        and head bounding boxes (if available)
        :return: pd.Dataframe with gaze information
        """
        raise MethodNotImplementedException("MethodNotImplementedException")

    def getAudioInput(self) -> pd.DataFrame:
        """
        Extracts Audio data for a single session based on given run config.
        Audio data consist of featurized audio in frequency buckets, and MFCC features (if available)
        :return: pd.Dataframe with audio information
        """
        raise MethodNotImplementedException("MethodNotImplementedException")

    def getMetaInput(self) -> dict:
        """
        Extracts meta information (required by some modules) for given session, including classroom object positioning,
        camera configs (if data is not captured from 3D sensing but by a camera in classrooms), etc.
        :return: dict with meta information
        """
        raise MethodNotImplementedException("MethodNotImplementedException")



class MethodNotImplementedException(Exception):
    """Raised when an information is expected but not provided by current driver"""
    def __init__(self, message='Key Empty'):
        # Call the base class constructor with the parameters it needs
        super(MethodNotImplementedException, self).__init__(message)
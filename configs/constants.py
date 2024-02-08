"""
Author: Anonymized
Created: Anonymized
Contains classes defining constants needed for session analytics
"""

from enum import Enum

class Constants:

    """Constants for different paths to be used"""

    #Pipeline Version
    ANALYTICS_PIPELINE_VERSION = '1.0.0'

    # Path where we store the logs
    LOG_DIR = 'cache/logs'
    LOG_FILE = 'videoOnly_analytics_engine_multiple_run.log'

    # Seconds in 1 minute
    SECS_IN_MIN = 60

    # Seconds in 1 hour
    SECS_IN_HR = 3600

    # Seconds in 1 day
    SECS_IN_DAY = 86400

    # milli, micro and nano seconds
    MILLISECS_IN_SEC = 1e3
    MICROSECS_IN_SEC = 1e6
    NANOSECS_IN_SEC = 1e9

    # days in 1 year
    DAYS_IN_YEAR = 365

    # No. of seconds for block level analysis
    BLOCK_SIZE = 120

    # Video resolution
    FULL_RES_X_MAX = 4096
    FULL_RES_Y_MAX = 2160

    # Default output dir for file posting
    DEFAULT_OUTPUT_DIR = 'cache/output'

# exit status from all across pipeline
class exitStatus(Enum):
    SUCCESS = 1
    FAILURE = 2
    PARTIAL_SUCCESS=3
    RESULTS_CACHED = 4
    NO_STUDENT_GAZE_DATA = 5
    NO_INSTRUCTOR_GAZE_DATA = 6
    NO_STUDENT_LOCATION_DATA = 7
    NO_INSTRUCTOR_LOCATION_DATA = 8
    NO_AUDIO_DATA = 9


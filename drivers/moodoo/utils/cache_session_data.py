"""
This utility function helps in writing and accessing cached data.
"""

from datetime import datetime
import os
import pickle

from configs.constants import exitStatus
from utils.time_utils import time_diff


def cache_session_data(session_input, logger):
    """
    Cache complete session input object for each seesion in pickle format

    Parameters:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        logger(logger)                 : The parent logger to log in function

    Returns:
        cache_status(exitStatus)       : Exit flag for caching success
    """
    t_cache_data_start = datetime.now()

    cache_dir = session_input.get('cache_dir')
    server_name = 'moodoo_files'
    session_id = session_input.get('session_id')

    cache_status = exitStatus.SUCCESS
    try:
        if not os.path.exists(f"{cache_dir}/{server_name}"):
            os.makedirs(f"{cache_dir}/{server_name}")
        with open(f"{cache_dir}/{server_name}/{session_id}.pb", "wb") as f:
            pickle.dump(session_input, f)
    except:

        cache_status = exitStatus.FAILURE

    t_cache_data_end = datetime.now()

    logger.info("Caching Session Input took | %.3f secs.", time_diff(t_cache_data_start, t_cache_data_end))

    return cache_status

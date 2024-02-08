"""
This files consists of common preprocessing for audio data in edulyze. it includes
1. Adding block_id for all frames

Author: Anonymized
Created at: Anonymized
"""

# basic libraries
import numpy as np
import pandas as pd

# custom libraries
from configs.constants import Constants


def preprocess_audio_data(df_raw_audio_data, raw_meta_data_dict, logger):
    session_start_timestamp = raw_meta_data_dict['session_start_timestamp']
    df_processed_audio_data = df_raw_audio_data
    df_processed_audio_data['block_id'] = ((df_processed_audio_data['timestamp'] / Constants.MILLISECS_IN_SEC)
                                              - session_start_timestamp) // Constants.BLOCK_SIZE
    df_processed_audio_data['block_id'] = df_processed_audio_data['block_id'].astype(int)
    audio_preprocessing_metrics = dict()

    return df_processed_audio_data, audio_preprocessing_metrics

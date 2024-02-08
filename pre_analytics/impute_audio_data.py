"""
Author: Anonymized
Created: Anonymized

This file contains functions imputes missing audio input data using various techniques.
"""

def impute_audio_data(session_input_object, processed_audio_data, logger_pass):
    """
    Wrapper function to imputes various audio features for processed audio data

    Parameters:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        processed_audio_data(dict)     : Dictionary containing audio data processed for analytics engine
        logger_pass(logger)            : The parent logger from which to derive the logger for the engine

    Returns:
        imputed_audio_data(dict)      : Dictionary containing audio data with missing values imputed for analytics
        imputation_metrics(dict)      : Dictionary containing information about characterstics of imputation performed
    """

    return processed_audio_data, {}

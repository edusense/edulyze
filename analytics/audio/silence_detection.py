"""
Author: Anonymized
Created: Anonymized

This file contains code to detect silences in classes at second level, and create statistics at block level.
"""

# Import python library functions
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import librosa
from sklearn.cluster import KMeans
import cv2
import matplotlib.pyplot as plt

# Import project level functions and classes

from configs.constants import Constants, exitStatus
from utils.time_utils import time_diff
from analytics.audio.utils import show_spectogram, get_peaks, audio_config


def run_silence_detection_module(session_input_object, session_output_object, logger_pass):
    """
    This is main function to run silence detection modules

    Parameters:
        session_input_object(dict)     : Dictionary containing all required inputs for session analytics
        session_output_object(dict)    : Dictionary to collect all outputs for session analytics
        logger_pass(logger)            : The parent logger from which to derive the logger for this function

    Returns:
        session_output_object(dict)    : Dictionary to collect all outputs for session analytics
    """
    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('silence_detection_module')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_silence_detection_start = datetime.now()

    # Initialize silence detection module

    silence_detection_results = {
        'second': dict(),
        'block': dict(),
        'session': dict(),
        'debug':dict()
    }

    run_status = exitStatus.SUCCESS

    # extract spectrogram, timestamps and block_id from audio input

    audio_input_df = session_input_object.get('input_audio_df', None)
    if audio_input_df is not None:
        audio_input = audio_input_df[audio_input_df.channel=='instructor'].to_dict(orient='records')
        audio_input = {xr['frameNumber']:xr for xr in audio_input}
    else:
        t_silence_detection_end = datetime.now()
        run_status = exitStatus.NO_AUDIO_DATA
        logger.info("Unable to run Silence detection module | %.3f secs.",
                    time_diff(t_silence_detection_start, t_silence_detection_end))

        return silence_detection_results, run_status

    mel_frequency_input = [np.array(audio_input[frameNumber]['melFrequency']) for frameNumber in audio_input.keys()]
    block_ids = np.array(
        [np.full(mel_frequency_input[i].shape[1], audio_input[frameNumber]['block_id']) for i, frameNumber in
         enumerate(audio_input.keys())]).flatten()
    timestamp_vals = np.array(
        [np.full(mel_frequency_input[i].shape[1], audio_input[frameNumber]['timestamp']) for i, frameNumber in
         enumerate(audio_input.keys())]).flatten()
    second_vals = timestamp_vals // Constants.MILLISECS_IN_SEC
    is_silence_detected = np.zeros_like(second_vals)

    # Create complete session spectogram from mel frequency input
    session_spectrogram = np.concatenate(mel_frequency_input, axis=1)
    samples_per_sec = mel_frequency_input[0].shape[1]
    loudness_spectrogram = librosa.power_to_db(session_spectrogram, ref=np.max)

    if mel_frequency_input[0] is None:
        logger.error("Mel frequency input not available, exiting module..")
        return silence_detection_results, exitStatus.FAILURE

    # preprocess spectogram to dataframe

    spect_index = np.arange(0, timedelta(seconds=loudness_spectrogram.shape[1] / mel_frequency_input[0].shape[1]),
                            timedelta(seconds=1 / mel_frequency_input[0].shape[1]))[-loudness_spectrogram.shape[1]:]
    spec_columns = librosa.mel_frequencies(n_mels=128)
    logger.debug(f"Shape of extracted session spectrogram:{spect_index.shape}, {spec_columns.shape}")
    df_spect = pd.DataFrame(loudness_spectrogram.T, columns=spec_columns, index=spect_index)

    # Detect silences at second level, looping over all blocks

    for block_id in range(np.max(block_ids) + 1):
        block_idx = np.where(block_ids==block_id)[0]
        is_silence_block = detect_coarse_silence_periods(df_spect.iloc[block_idx], samples_per_sec, logger)
        is_silence_detected[block_idx] = is_silence_block

    # Fill second, block and session level results

    df_silence = pd.DataFrame(np.array([block_ids,second_vals,is_silence_detected]).T,columns=['block_id','second','is_silence'])
    silence_detection_results['debug']['is_silence_df'] = df_silence

    df_second_silence = df_silence.groupby(['block_id','second'], as_index=False)['is_silence'].agg(pd.Series.mode)
    silence_detection_results['second']['silence_seconds'] = df_second_silence

    df_block_fraction = df_silence.groupby(['block_id'],as_index=False).agg({'is_silence':['count','sum']})
    df_block_fraction['block_fraction'] = df_block_fraction['is_silence']['sum'] / df_block_fraction['is_silence']['count']
    silence_detection_results['block']['silence_block_fraction'] = df_block_fraction[['block_id','block_fraction']]

    session_fraction = df_silence['is_silence'].sum() / df_silence.shape[0]
    silence_detection_results['session']['silence_session_fraction'] = session_fraction

    t_silence_detection_end = datetime.now()

    logger.info("Silence detection module took | %.3f secs.",
                time_diff(t_silence_detection_start, t_silence_detection_end))

    return silence_detection_results, run_status


def detect_coarse_silence_periods(df_spectrogram, samples_per_sec, logger, debug_plot=False):
    """
    This function detects coarse silence periods to run silence detection modules

    Parameters:
        df_spectrogram(pd.DataFrame)   : spectrogram to detect silences from
        samples_per_sec(int)           : Count of samples per second in spectrogram
        logger(logger)                 : The parent logger
        debug_plot(bool)               : Whether we run it in debug mode or not

    Returns:
        is_silence(np.ndarray)         : An array marking each time section as silence/not silence
    """

    # Initialize input from audio config

    FILTER_ROW_SIZE = int(samples_per_sec / 2)

    FREQUENCY_MIN = audio_config.get('FREQUENCY_MIN')
    FREQUENCY_MAX = audio_config.get('FREQUENCY_MAX')
    NUM_CLUSTERS = audio_config.get("NUM_CLUSTERS")

    COARSE_SILENCE_WINDOW_IN_SEC = audio_config.get('COARSE_SILENCE_WINDOW_IN_SEC')
    DISTANCE_WIDTH_RATIO = audio_config.get('DISTANCE_WIDTH_RATIO')

    start_min = df_spectrogram.index.min().seconds // 60 % 60
    end_min = df_spectrogram.index.max().seconds // 60 % 60

    if end_min == start_min:
        end_min += 1

    # Min Max Filtering for raw spectrogram

    min_max_kernel = np.ones((FILTER_ROW_SIZE, 5), np.uint8)
    df_min_filtered = df_spectrogram.copy(deep=True)
    df_min_filtered.iloc[:, :] = cv2.dilate(df_spectrogram.values, min_max_kernel)
    df_minmax_filtered = df_min_filtered.copy(deep=True)
    df_minmax_filtered.iloc[:, :] = cv2.erode(df_minmax_filtered.values, min_max_kernel)

    if debug_plot:
        show_spectogram(df_spectrogram, start_min, end_min, title="Raw Spectrogram")
        show_spectogram(df_min_filtered, start_min, end_min, title="Max Filtered")
        show_spectogram(df_minmax_filtered, start_min, end_min, title="Minmax Filtered")

    # Kmeans Clustering for filtered spectrogram

    filtered_columns = [xr for xr in df_spectrogram.columns if ((xr <= FREQUENCY_MAX) & (xr >= FREQUENCY_MIN))]
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0).fit(df_min_filtered[filtered_columns].values)
    min_cluster = np.argmin(np.sum(kmeans.cluster_centers_, axis=1))

    if debug_plot:
        plt.figure(figsize=(20, 10))
        plt.scatter(range(len(kmeans.labels_)), kmeans.labels_)
        plt.figure(figsize=(20, 10))
        for i in range(NUM_CLUSTERS):
            plt.plot(kmeans.cluster_centers_[i], label=i)
        plt.title(f"Num Clusters:{NUM_CLUSTERS}")
        plt.legend()

    # Post processing detected silence periods

    is_min_cluster = np.zeros(kmeans.labels_.size)
    is_min_cluster[kmeans.labels_ == min_cluster] = 1

    # Merge Neighbouring peaks

    sufficient_peak_distance = samples_per_sec
    sufficient_peak_width = samples_per_sec

    peak_starts, peak_ends = get_peaks(is_min_cluster)
    peak_widths, peak_distances = peak_ends - peak_starts, peak_starts[1:] - peak_ends[:-1]
    peak_distances = np.insert(peak_distances, 0, peak_starts[0])
    merged_peaks = np.copy(is_min_cluster)

    for i in range(1, peak_starts.shape[0]):
        if peak_distances[i] < sufficient_peak_distance:
            max_peak_width = max(peak_widths[i], peak_widths[i - 1])
            if (max_peak_width > sufficient_peak_width) & (max_peak_width > DISTANCE_WIDTH_RATIO * peak_distances[i]):
                merged_peaks[peak_starts[i - 1]:peak_ends[i] + 1] = 1
                # update peak_width for last peak
                peak_widths[i] = peak_ends[i] - peak_starts[i - 1]
        elif peak_widths[i] >= COARSE_SILENCE_WINDOW_IN_SEC * samples_per_sec:
            merged_peaks[peak_starts[i]:peak_ends[i] + 1] = 1

    # Remove Small/Insignificant Peaks

    peak_starts, peak_ends = get_peaks(merged_peaks)
    peak_widths, peak_distances = peak_ends - peak_starts, peak_starts[1:] - peak_ends[:-1]
    peak_distances = np.insert(peak_distances, 0, peak_starts[0])

    for i in range(1, peak_starts.shape[0]):
        if peak_widths[i] <= sufficient_peak_width:
            merged_peaks[peak_starts[i]:peak_ends[i] + 1] = 0

    if debug_plot:
        plt.figure(figsize=(20, 10))
        plt.plot(is_min_cluster)
        plt.title("Min Clusters Detected")
        plt.figure(figsize=(20, 10))
        plt.plot(merged_peaks)
        plt.title("Merged Peaks")

    # df_silence_spect = df_spectrogram.copy(deep=True)
    # df_silence_spect.iloc[merged_peaks == 1, :] = 0
    # if debug_plot:
    #     show_spectogram(df_spectrogram, start_min, end_min, title="Raw Spectrogram")
    #     show_spectogram(df_silence_spect, start_min, end_min, title="Detected Silence Periods")
    #
    # # return results in terms of silence periods, fraction of silences
    #
    # silence_periods = []
    # final_peak_starts, final_peak_ends = get_peaks(merged_peaks)
    # for peak_start, peak_end in zip(final_peak_starts, final_peak_ends):
    #     silence_periods.append((str(df_silence_spect.index[peak_start]), str(df_silence_spect.index[peak_end - 1])))

    return merged_peaks

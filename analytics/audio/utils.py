"""
Author: Anonymized
Created: Anonymized

This file contains all utility function for audio modality
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
import datetime
import base64
from PIL import Image
from io import BytesIO as _BytesIO
import requests
import json
import pickle
import collections
import librosa, librosa.display
from scipy import signal
import sklearn

audio_config = {
    # Silence detection configs
    'FREQUENCY_MIN': 200,
    'FREQUENCY_MAX': 2000,
    'NUM_CLUSTERS': 4,

    'COARSE_SILENCE_WINDOW_IN_SEC': 3,
    'DISTANCE_WIDTH_RATIO': 0.2,

    # Object detection configs
    'NORM_DIFF_THRESHOLD': 0.2,
    'OBJECT_NOISE_FEATURES': {
        'mfcc': [0, 1, 2, 3, 4],
        'poly': []
    }
}


def show_spectogram(df, start_min, end_min, x_label_count=60, title=None):
    """
    Shows spectogram from start_min to end min for debugging purposes
    """
    plt.figure(figsize=(20, 10))
    df_show = df[datetime.timedelta(minutes=start_min):datetime.timedelta(minutes=end_min)]
    div_per_secs = df_show.shape[0] / ((end_min - start_min) * 60)
    secs_per_label = end_min - start_min
    librosa.display.specshow(df_show.values.T, sr=32000, y_axis='mel', fmax=8000, hop_length=2000)
    plt.xticks(np.arange(0, df_show.shape[0] + 1, div_per_secs * secs_per_label),
               np.arange(0, ((end_min - start_min) * 60) + 1, secs_per_label), rotation=45)
    plt.xlabel('Time (in secs)', fontsize=20)
    plt.ylabel('Hz', fontsize=20)
    if title is not None:
        plt.title(title, fontsize=24)
    plt.colorbar(format='%+2.0f', label='loudness(in dBs)')
    plt.show()

    return


def get_peaks(peak_signal):
    """
    Get peak starts and peak end from a peak signal
    """
    peak_starts = np.where(peak_signal[1:] > peak_signal[:-1])[0]
    peak_ends = np.where(peak_signal[1:] < peak_signal[:-1])[0]

    if (peak_signal[0]):
        peak_starts = np.insert(peak_starts, 0, 0)
    if (peak_signal[-1]):
        peak_ends = np.insert(peak_ends, peak_ends.shape[0], peak_signal.shape[0])

    return peak_starts, peak_ends

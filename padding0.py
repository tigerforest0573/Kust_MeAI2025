# Original work Copyright 2026 Zijie Zhu . <https://github.com/tigerforest0573/Kust_MeAI2025>
import numpy as np
from helper_code import *
from scipy.signal import resample
from typing import Any, Dict, List, Optional, Tuple, Union
class Resample:
    """Resample the input sequence.
    """
    def __init__(self,
                 target_length: Optional[int] = None,
                 target_fs: Optional[int] = None) -> None:
        self.target_length = target_length
        self.target_fs = target_fs

    def __call__(self, x: np.ndarray, fs: Optional[int] = None) -> np.ndarray:
        if fs and self.target_fs and fs != self.target_fs:
            x = resample(x, int(x.shape[1] * self.target_fs / fs), axis=1)
        elif self.target_length and x.shape[1] != self.target_length:
            x = resample(x, self.target_length, axis=1)
        return x
################################################################################
def get_12ECG_features(header, recording,useFilter=True):
    featureIndividual = []
    sample_Fs = get_sampling_frequency(header)
    seglength = int(get_num_samples(header)) // sample_Fs
    target_len = int(10 * sample_Fs)
    if recording.shape[0] > recording.shape[1]:
            recording = recording.T  # 转为 (n_leads, n_samples)
            print(f"转置后Shape: {recording.shape}")  #(样本数, 导联数)
    for i in range(12):
        sig = recording[i, :]
        res = dataLenCheck(sig[:target_len], target_len)  # 10 s    res = dataLenCheck(down_filtered_ecg_lead[:1000], 1000)

        featureIndividual.append(res) #list
    feature_array = np.array(featureIndividual,dtype='float32')

    target_fs = 100
    if sample_Fs != target_fs:
        new_length = int(feature_array.shape[1] * target_fs / sample_Fs)
        feature_array = resample(feature_array, new_length, axis=1)
        feature_array = feature_array.astype(np.float32)

    return feature_array

################################################################################
# 数据截断与长度统一
################################################################################
# def dataLenCheck(data, window):
#     if len(data) == window:
#         return data
#     elif len(data) < window:
#         data += [0 for i in range(window - len(data))]  # Fill data by 0.
#         return data
#     else:
#         raise Exception("Error in dataLenCheck")
#     # 防止数据片段小于window

def dataLenCheck(data, window):
    current_len = len(data)
    if current_len == window:
        return data
    elif current_len < window:
        pad_width = window - current_len
        padded_data = np.pad(data, (0, pad_width), mode='constant', constant_values=0)
        return padded_data
    else:
        raise Exception("Error in dataLenCheck: Data is longer than window")

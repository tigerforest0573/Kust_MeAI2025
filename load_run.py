import os
import numpy as np
import pandas as pd
from helper_code import *
from padding0 import get_12ECG_features
from tqdm import tqdm

def load_record_feature(record_path):
    """
    从单条记录路径中提取 ECG 特征，推理时使用
    返回值: ndarray, shape [12, 1000]
    """
    try:
        signal, _ = load_signals(record_path)           # [12, N]
        header = load_header(record_path)
        features = get_12ECG_features(header, signal)   # [12, 1000]，滤波+归一化
        return features
    except Exception as e:
        raise ValueError(f"[ERROR]Failed to load: {record_path}\nReason: {e}")




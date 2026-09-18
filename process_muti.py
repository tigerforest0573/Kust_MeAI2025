# Original work Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>
# Modified work Copyright 2026 Zijie Zhu. <https://github.com/tigerforest0573/xxx>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import concurrent.futures 
from functools import partial 

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm


_LEAD_NAMES = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def get_parser():
    description = "Process WFDB ECG database."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i',
                        '--input_dir',
                        type=str,
                        required=True,
                        help="Path to the WFDB ECG database directory.")
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        required=True,
                        help="Path to the directory where the preprocessed signals will be saved.")
    parser.add_argument('--index_path',
                        type=str,
                        default="./index_origin.csv",
                        help="Path to the index file.")
    parser.add_argument('--num_workers', #指定线程数
                        type=int,
                        default=os.cpu_count(),
                        help="Number of parallel workers.")
    args = parser.parse_args()
    return args


def find_records(root_dir):
    """Find all the .hea files in the root directory and its subdirectories.
    Args:
        root_dir (str): The directory to search for .hea files.
    Returns:
        records (set): A set of record names.
                        (e.g., ['database/1/ecg001', 'database/1/ecg001', ..., 'database/9/ecg991'])
    """
    records = set()
    for root, _, files in os.walk(root_dir):
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension == '.hea':
                record = os.path.relpath(os.path.join(root, file), root_dir)[:-4]
                records.add(record)
    records = sorted(records)
    return records


def moving_window_crop(x: np.ndarray, crop_length: int, crop_stride: int) -> np.ndarray:
    """Crop the input sequence with a moving window.
    """
    if crop_length > x.shape[1]:
        raise ValueError(f"crop_length must be smaller than the length of x ({x.shape[1]}).")
    start_idx = np.arange(0, x.shape[1] - crop_length + 1, crop_stride)
    return [x[:, i:i + crop_length] for i in start_idx]



def process_record(record_rel_path, input_dir, output_dir):
    try:
        record_rel_dir, record_name = os.path.split(record_rel_path)
        save_dir = os.path.join(output_dir, record_rel_dir)
        os.makedirs(save_dir, exist_ok=True)
        source_name = record_rel_dir.split("/")[0]

        # 读取数据
        signal, record_info = wfdb.rdsamp(os.path.join(input_dir, record_rel_path))
        binary_label = 0 
        # 获取头文件里的注释部分
        comments = record_info.get('comments', [])
        for comment in comments:
            #检查是否包含目标字段 (忽略大小写)
            if "Chagas label" in comment:
                #提取冒号后的值，去除空格
                
                val_str = comment.split(":")[-1].strip().lower()
                
                if val_str == 'true':
                    binary_label = 1
                else:
                    binary_label = 0
                break 
        CAP_lead = [name.upper() for name in record_info["sig_name"]]
        lead_idx = np.array([CAP_lead.index(lead_name) for lead_name in _LEAD_NAMES])
        signal = signal[:, lead_idx]
        fs = record_info["fs"]
        signal_length = record_info["sig_len"]
        target_len = 10 * fs
        
        # Padding
        if signal_length < target_len:
            pad_len = target_len - signal_length
            signal = np.pad(signal, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
            
        # Cropping
        cropped_signals = moving_window_crop(signal.T, crop_length=10 * fs, crop_stride=10 * fs)
        
        processed_rows = []
        for idx, cropped_signal in enumerate(cropped_signals):
            if cropped_signal.shape[1] != 10 * fs or np.isnan(cropped_signal).any():
                continue
            
            file_name = f"{record_name}_{idx}.pkl"
            save_path = os.path.join(save_dir, file_name)
            
            # 保存 pickle
            pd.to_pickle(cropped_signal.astype(np.float32), save_path)
            
            # 收集索引信息
            processed_rows.append({
                "RELATIVE_FILE_PATH": f"{record_rel_path}_{idx}.pkl",
                "FILE_NAME": file_name,
                "SAMPLE_RATE": fs,
                "SOURCE": source_name,
                "LABEL": binary_label
            })
            
        return processed_rows
    except Exception as e:
        print(f"Error processing {record_rel_path}: {e}")
        return []


def run(args):
    # Identify the header files
    record_rel_paths = find_records(args.input_dir)
    print(f"Found {len(record_rel_paths)} records.")

    #并行处理
    all_rows = []
    
    worker_func = partial(process_record, input_dir=args.input_dir, output_dir=args.output_dir)

    print(f"Processing with {args.num_workers} workers...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        #tqdm
        results = list(tqdm(executor.map(worker_func, record_rel_paths), total=len(record_rel_paths)))
    
    # 展平结果列表
    for res in results:
        all_rows.extend(res)

    #构建 DataFrame
    index_df = pd.DataFrame(all_rows, columns=["RELATIVE_FILE_PATH", "FILE_NAME", "SAMPLE_RATE", "SOURCE","LABEL"])
    
    print(f"Saved {len(index_df)} cropped signals.")
    os.makedirs(os.path.dirname(args.index_path), exist_ok=True)
    index_df.to_csv(args.index_path, index=False)


if __name__ == "__main__":
    run(get_parser())
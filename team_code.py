#!/usr/bin/env python
# Modified work Copyright 2026 Zijie Zhu. <https://github.com/tigerforest0573/Kust_MeAI2025>

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys

from helper_code import *
import argparse
from process_muti import run
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from main_downstream import main, parse
import models.encoder as encoder
import yaml
import torch
import util.transforms as T
import torch.nn.functional as F
from load_run import load_record_feature

from util.transforms import get_transforms_from_config, get_rand_augment_from_config
################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')
    combine_model_parts()
    index_gen_args = argparse.Namespace(
        input_dir=data_folder,
        output_dir=model_folder,
        index_path=os.path.join(model_folder, "index_origin.csv"),
        num_workers=os.cpu_count()  
    )
    run(index_gen_args)
    index_path_origin = os.path.join(model_folder, "index_origin.csv")
    df = pd.read_csv(index_path_origin)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df['fold'] = -1
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df['LABEL'])):
        df.loc[val_idx, 'fold'] = fold_idx
    index_path_fold = os.path.join(model_folder, "index_fold.csv")
    df.to_csv(index_path_fold, index=False)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path1 = os.path.join(current_dir, 'updated_labels2.csv')
    path2 = index_path_fold
    try:
        df1 = pd.read_csv(path1)
        df2 = pd.read_csv(path2)
        print("成功读取文件！")
    except FileNotFoundError as e:
        print(f"文件未找到，请检查路径是否正确：\n{e}")
        exit()
    df2['temp_match_id'] = df2['FILE_NAME'].apply(lambda x: int(str(x).split('_')[0]))
    merged_df = df2.merge(df1[['exam_id', 'new_label']], 
                        left_on='temp_match_id', 
                        right_on='exam_id', 
                        how='left')
    merged_df['LABEL'] = merged_df['new_label'].fillna(merged_df['LABEL'])
    final_df = merged_df.drop(columns=['temp_match_id', 'exam_id', 'new_label'])

    print("合并后的数据：")
    print(final_df.head(20))
    matched_count = merged_df['new_label'].notna().sum()
    print(f"共 {len(df2)} 行， {matched_count} 行已更新，{len(df2) - matched_count} 行保留原值")
    index_path_final= os.path.join(model_folder, "index_final.csv")
    final_df.to_csv(index_path_final, index=False)
    config_ymal_path = os.path.join(current_dir, 'configs/downstream/st_mem.yaml')
    target_config_path = os.path.join(model_folder, 'st_mem_runtime.yaml')
    with open(config_ymal_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    config_dict['output_dir'] = model_folder
    config_dict['dataset']['index_dir'] = model_folder
    config_dict['dataset']['ecg_dir'] = model_folder
    config_dict['dataset']['all_data_csv'] = os.path.join(model_folder, 'index_final.csv')
    with open(target_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    # Train the models on the features.
    if verbose:
        print('Training the model on the data...')

    # This very simple model trains a random forest model with very simple features.
    encoder1_path = os.path.join(current_dir, 'encoder.pth')
    sys.argv = [
        'main_downstream.py',             
        '--config_path', target_config_path,     
        '--output_dir', model_folder,    
        '--encoder_path', encoder1_path,   
        '--exp_name', 'challenge_test'   
    ]
    down_config = parse()
    main(down_config)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
GLOBAL_CONFIG = None
def load_model(model_folder, verbose):
    global GLOBAL_CONFIG
    config_path = os.path.join(model_folder, 'st_mem_runtime.yaml')
    model_path = os.path.join(model_folder, 'challenge_test', 'best-loss.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        GLOBAL_CONFIG = config
    model_name = config['model_name']
    model = encoder.__dict__[model_name](**config['model'])
    print(f"正在从 {model_path} 加载权重")
    print(device)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    print(model)
    msg = model.load_state_dict(checkpoint['model'], strict=True)
    print(f"权重加载: {msg}")
    print(GLOBAL_CONFIG)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    # Load the model.
    print(GLOBAL_CONFIG)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    signal = load_record_feature(record)

    transforms = get_transforms_from_config(GLOBAL_CONFIG["dataset"]["eval_transforms"])
    transforms = T.Compose(transforms + [T.ToTensor()])
    processed_signal = transforms(signal)
    #(12, 1000) -> (1, 12, 1000)
    processed_signal = np.array(processed_signal, dtype=np.float32)       
    #转换为 Tensor并加 batch维度
    input_tensor = torch.from_numpy(processed_signal).unsqueeze(0).to(device)
    
    logits = model(input_tensor)
    probs = F.softmax(logits, dim=1)
    pred_class = torch.argmax(probs, dim=1)
    return pred_class.cpu().item(), probs.detach().cpu().numpy()[0, 1]

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################
def combine_model_parts(output_file='encoder.pth', part_prefix='encoder.pth.part'):
    path = os.path.dirname(__file__) 
    parts = sorted([f for f in os.listdir(path) if f.startswith(part_prefix)])
    if not parts:
        print("错误：未找到模型分卷文件！")
        return
    print(f"正在合并 {len(parts)} 个分卷文件...")
    with open(os.path.join(path, output_file), 'wb') as output:
        for part in parts:
            with open(os.path.join(path, part), 'rb') as f:
                output.write(f.read())
    print(f"合并完成：{output_file}")
# Extract your features.
def extract_features(record):
    header = load_header(record)

    # Extract the age from the record.
    age = get_age(header)
    age = np.array([age])

    # Extract the sex from the record and represent it as a one-hot encoded vector.
    sex = get_sex(header)
    sex_one_hot_encoding = np.zeros(3, dtype=bool)
    if sex.casefold().startswith('f'):
        sex_one_hot_encoding[0] = 1
    elif sex.casefold().startswith('m'):
        sex_one_hot_encoding[1] = 1
    else:
        sex_one_hot_encoding[2] = 1

    # Extract the source from the record (but do not use it as a feature).
    source = get_source(header)

    # Load the signal data and fields. Try fields.keys() to see the fields, e.g., fields['fs'] is the sampling frequency.
    signal, fields = load_signals(record)
    channels = fields['sig_name']

    # Reorder the channels in case they are in a different order in the signal data.
    reference_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    num_channels = len(reference_channels)
    signal = reorder_signal(signal, channels, reference_channels)

    # Compute two per-channel features as examples.
    signal_mean = np.zeros(num_channels)
    signal_std = np.zeros(num_channels)

    for i in range(num_channels):
        num_finite_samples = np.sum(np.isfinite(signal[:, i]))
        if num_finite_samples > 0:
            signal_mean[i] = np.nanmean(signal)
        else:
            signal_mean = 0.0
        if num_finite_samples > 1:
            signal_std[i] = np.nanstd(signal)
        else:
            signal_std = 0.0

    # Return the features.

    return age, sex_one_hot_encoding, source, signal_mean, signal_std

# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)
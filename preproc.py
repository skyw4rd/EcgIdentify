import argparse
import random
import numpy as np
import os
import re

from typing import List
from pathlib import Path

import wfdb
import neurokit2 as nk
from pyts.image import GramianAngularField
from pyts.approximation import PiecewiseAggregateApproximation


# ecg信号预处理函数
def get_rpeaks(sig, sampling_rate):
    _, rpeaks = nk.ecg_peaks(sig, sampling_rate=sampling_rate)
    return rpeaks['ECG_R_Peaks']

def ecg_correlation(ecg1, ecg2):
    if len(ecg1) != len(ecg2):
        raise ValueError("signals must be same length")
    
    mean1 = np.mean(ecg1)
    mean2 = np.mean(ecg2)
    
    covariance = np.sum((ecg1 - mean1) * (ecg2 - mean2))
    std1 = np.sqrt(np.sum((ecg1 - mean1) ** 2))
    std2 = np.sqrt(np.sum((ecg2 - mean2) ** 2))
    
    if std1 * std2 == 0:
        return 0.0
    else:
        return covariance / (std1 * std2)


def get_args():
    parser = argparse.ArgumentParser('ecg signal previous processing', add_help=False)
    parser.add_argument('--dataset-path', default='F:/summer/light_ecg_id/dataset/ecg-id', type=str)
    parser.add_argument('--segment-path', default='ecg_segment/', type=str)
    parser.add_argument('--sampling-rate', default=500, type=int)
    return parser.parse_args() 


def get_segment(clean_sig, heart_rate, rpeaks, sampling_rate):
    # heart_rate = np.mean(nk.signal_rate(rpeaks, sampling_rate=sampling_rate, desired_length=len(clean_sig)))
    # R峰前比例
    ratio_pre = 0.35
    window_size = 60 / heart_rate  # Beats  second
    epochs_start = ratio_pre * window_size
    epochs_end = (1 - ratio_pre) * window_size

    heartbeats = nk.epochs_create(clean_sig, 
                                  events=rpeaks, 
                                  epochs_start=-epochs_start, 
                                  epochs_end=epochs_end, 
                                  sampling_rate=sampling_rate)
    return heartbeats


def write_segment(rc_pt, save_path, heart_rate, sampling_rate):
    rc = wfdb.rdrecord(rc_pt, channels=[1])
    dt = rc.p_signal.flatten()
    cl_sig = nk.ecg_clean(dt, sampling_rate=sampling_rate, method='hamilton2002')
    rpeaks = get_rpeaks(cl_sig, sampling_rate=sampling_rate)
    heart_beats = get_segment(cl_sig, heart_rate, rpeaks=rpeaks, sampling_rate=sampling_rate)
    df = nk.epochs_to_df(heart_beats)
    df_pivoted = df.pivot(index="Time", columns="Label", values='Signal')
    median_heartbeat = df_pivoted.median(axis=1)
    
    del_cols = []
    for col in df_pivoted.columns:
        if ecg_correlation(df_pivoted[col], median_heartbeat) < 0.7:
            del_cols.append(col)
    df_pivoted = df_pivoted.drop(del_cols, axis=1)
    
    csv_save_path = save_path.joinpath(f'sample_{rc_pt.name}.csv')
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
    df_pivoted.to_csv(csv_save_path)

def get_heartrate(dataset_path: Path, sampling_rate: int) -> List:
    heart_rates = []
    for person_dir in dataset_path.iterdir():
        if person_dir.is_dir():
            # person_name = person_dir.name
            rec_lst = ['rec_1', 'rec_2']
            s = 0
            for rec in rec_lst:
                rec_path = person_dir.joinpath(rec)
                rc = wfdb.rdrecord(rec_path, channels=[1])
                dt = rc.p_signal.flatten()
                cl_sig = nk.ecg_clean(dt, sampling_rate=sampling_rate, method='hamilton2002')
                rpeaks = get_rpeaks(cl_sig, sampling_rate=sampling_rate)
                heart_rate = np.mean(nk.signal_rate(rpeaks, sampling_rate=sampling_rate, desired_length=len(cl_sig)))
                s += heart_rate
            heart_rates.append(s / 2)
    return heart_rates

def gen_samples(args):
    dataset_path = Path(args.dataset_path)
    segment_path = Path(args.segment_path)
    sampling_rate = args.sampling_rate
    # 计算平均心率

    heart_rates = get_heartrate(dataset_path=dataset_path, sampling_rate=sampling_rate)

    for idx, person_dir in enumerate(dataset_path.iterdir()):
        if person_dir.is_dir():
            person_name = person_dir.name
            rec_lst = []
            for rec_file in person_dir.iterdir():
                if re.search('rec', rec_file.stem) and rec_file.stem not in rec_lst:
                    rec_lst.append(rec_file.stem)
            for rec in rec_lst:
                rec_path = person_dir.joinpath(rec)
                save_path = segment_path.joinpath(person_name)
                write_segment(rec_path, save_path, heart_rates[idx])

def main(args): 
    # 生成数据样本
    gen_samples(args)
    return 

if __name__ == '__main__':
    args = get_args()
    main(args)
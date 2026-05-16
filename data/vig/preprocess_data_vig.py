import os

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt

# ======================================================
# 参数设置
# ======================================================
fs = 200
trial_sec = 8
trial_len = fs * trial_sec  # 1600
expected_trials = 885

eeg_dir = r"E:/SEED-VIG/Raw_Data"
label_dir = r"E:/SEED-VIG/perclos_labels"

save_eeg_dir = r"E:/SEED-VIG/processed/processed"
save_label_dir = r"E:/SEED-VIG/processed/labels"

os.makedirs(save_eeg_dir, exist_ok=True)
os.makedirs(save_label_dir, exist_ok=True)


# ======================================================
# 带通滤波
# ======================================================
def bandpass_filter(data, lowcut=1, highcut=50, fs=200, order=4):
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return filtfilt(b, a, data, axis=0)


# ======================================================
# z-score 归一化
# ======================================================
def z_score_normalize(data):
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    return (data - mean) / (std + 1e-8)


# ======================================================
# 主处理函数
# ======================================================
def process_subject(eeg_path, label_path):
    subject = os.path.splitext(os.path.basename(eeg_path))[0]
    print(f"\n📂 正在处理被试: {subject}")

    # ================= EEG =================
    eeg_mat = sio.loadmat(eeg_path)
    EEG_data = eeg_mat["EEG"]["data"][0, 0]

    if EEG_data.shape[0] == 17:
        EEG_data = EEG_data.T  # (time, channel)

    print(f"  EEG 原始形状: {EEG_data.shape}")

    filtered = bandpass_filter(EEG_data, fs=fs)
    normalized = z_score_normalize(filtered)

    n_samples, n_channels = normalized.shape
    n_trials = n_samples // trial_len
    assert n_trials == expected_trials, f"EEG trial 数异常: {n_trials}"

    eeg_trials = normalized[: n_trials * trial_len]
    eeg_trials = eeg_trials.reshape(n_trials, trial_len, n_channels)
    eeg_trials = eeg_trials.transpose(0, 2, 1)  # (885,17,1600)

    # ================= LABEL =================
    label_mat = sio.loadmat(label_path)
    perclos = np.squeeze(label_mat["perclos"])
    assert len(perclos) == expected_trials, "Label 数量不匹配"

    labels = np.where(perclos < 0.35, 0, 1)

    # ================= 保存 =================
    eeg_save_path = os.path.join(save_eeg_dir, f"{subject}_EEG_preprocessed.npy")
    label_save_path = os.path.join(save_label_dir, f"{subject}_label.npy")

    np.save(eeg_save_path, eeg_trials)
    np.save(label_save_path, labels)

    awake = np.sum(labels == 0)
    fatigue = np.sum(labels == 1)

    print(f"  ✅ EEG 保存: {eeg_trials.shape}")
    print(f"  ✅ Label 保存: 清醒={awake}, 疲劳={fatigue}")


# ======================================================
# 批量执行
# ======================================================
eeg_files = sorted([f for f in os.listdir(eeg_dir) if f.endswith(".mat")])

for file in eeg_files:
    eeg_path = os.path.join(eeg_dir, file)
    label_path = os.path.join(label_dir, file)

    if not os.path.exists(label_path):
        print(f"⚠️ 缺少标签文件: {file}")
        continue

    process_subject(eeg_path, label_path)

folder = r"E:/SEED-VIG/processed/processed"

# 获取文件列表并排序
files = sorted(os.listdir(folder))

for idx, filename in enumerate(files, start=1):
    old_path = os.path.join(folder, filename)
    new_name = f"{idx}_{filename}"
    new_path = os.path.join(folder, new_name)
    os.rename(old_path, new_path)
    print(f"{filename} -> {new_name}")

print("✅ 文件重命名完成！")

folder = r"E:/SEED-VIG/processed/labels"

# 获取文件列表并排序
files = sorted(os.listdir(folder))

for idx, filename in enumerate(files, start=1):
    old_path = os.path.join(folder, filename)
    new_name = f"{idx}_{filename}"
    new_path = os.path.join(folder, new_name)
    os.rename(old_path, new_path)
    print(f"{filename} -> {new_name}")

print("✅ 文件重命名完成！")

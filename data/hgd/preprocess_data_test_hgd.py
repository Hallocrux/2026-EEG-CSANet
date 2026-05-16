import os
from collections import OrderedDict

import numpy as np
from hgd_data_utils import create_data_label_from_raw_mne, resample_raw_mne
from hgd_dataset import BBCIDataset
from hgd_signal_utils import exponential_running_standardize, highpass_cnt, mne_apply

data_path = r"D:/high-gamma-dataset/data/test"
save_path = r"process_data/test"
os.makedirs(save_path, exist_ok=True)

time_window = [0, 4000]  # epoch 时间窗口 ms
C_sensors = [
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "C3",
    "C4",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "FC3",
    "FCz",
    "FC4",
    "C5",
    "C1",
    "C2",
    "C6",
    "CP3",
    "CPz",
    "CP4",
    "FFC5h",
    "FFC3h",
    "FFC4h",
    "FFC6h",
    "FCC5h",
    "FCC3h",
    "FCC4h",
    "FCC6h",
    "CCP5h",
    "CCP3h",
    "CCP4h",
    "CCP6h",
    "CPP5h",
    "CPP3h",
    "CPP4h",
    "CPP6h",
    "FFC1h",
    "FFC2h",
    "FCC1h",
    "FCC2h",
    "CCP1h",
    "CCP2h",
    "CPP1h",
    "CPP2h",
]

marker_def = OrderedDict(
    [("Right Hand", [1]), ("Left Hand", [2]), ("Rest", [3]), ("Feet", [4])]
)

for sub_id in range(1, 15):
    print(f"Processing sub {sub_id}")
    data_file = os.path.join(data_path, f"{sub_id}.mat")

    loader = BBCIDataset(data_file)
    print("Loading data...")
    eeg_raw = loader.load()  # RawArray

    # 保留 EEG 通道 + 刺激通道
    eeg_raw = eeg_raw.pick_channels(C_sensors + ["STI 014"])
    # 设置刺激通道类型
    eeg_raw.set_channel_types({"STI 014": "stim"})

    print("Resampling...")
    eeg_raw = resample_raw_mne(eeg_raw, 250.0)

    print("Highpassing...")

    def highpass_eeg_only(data):
        new_data = data.copy()
        new_data[: len(C_sensors), :] = highpass_cnt(
            data[: len(C_sensors), :], 0, eeg_raw.info["sfreq"], filt_order=3, axis=1
        )
        return new_data

    eeg_raw = mne_apply(highpass_eeg_only, eeg_raw)

    print("Standardizing...")

    def standardize_eeg_only(data):
        new_data = data.copy()
        new_data[: len(C_sensors), :] = exponential_running_standardize(
            data[: len(C_sensors), :].T, factor_new=1e-3, init_block_size=1000, eps=1e-4
        ).T
        return new_data

    eeg_raw = mne_apply(standardize_eeg_only, eeg_raw)

    # 划分 epochs
    print("Creating trials...")
    train_data, train_label = create_data_label_from_raw_mne(
        eeg_raw, marker_def, time_window
    )

    # 剔除异常 trial
    clean_trial_mask = np.max(np.abs(train_data), axis=(1, 2)) < 800
    train_data = train_data[clean_trial_mask][:, :-1, :]
    train_label = train_label[clean_trial_mask]

    print(f"Data shape: {train_data.shape}")
    print(f"Label shape: {train_label.shape}")
    print(f"Clean trials: {np.sum(clean_trial_mask)} / {len(clean_trial_mask)}")

    # 保存
    np.save(os.path.join(save_path, f"{sub_id}_data.npy"), train_data)
    np.save(os.path.join(save_path, f"{sub_id}_label.npy"), train_label)

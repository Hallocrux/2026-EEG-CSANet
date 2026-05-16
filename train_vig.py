import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from data.dataset import eegDataset
from model.baseModel import baseModel

torch.set_num_threads(10)


def setRandom(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dictToYaml(filePath, dictToWrite):
    with open(filePath, "w", encoding="utf-8") as f:
        yaml.dump(dictToWrite, f, allow_unicode=True)


def load_SEEDVIG_data(root_path, subject_id):
    subject_id = str(subject_id)
    label_dir = os.path.join(root_path, "labels")
    eeg_dir = os.path.join(root_path, "processed")

    eeg_files = [f for f in os.listdir(eeg_dir) if f.endswith("EEG_preprocessed.npy")]
    label_files = [f for f in os.listdir(label_dir) if f.endswith("_label.npy")]

    eeg_file = [f for f in eeg_files if f.split("_")[0] == subject_id]
    label_file = [f for f in label_files if f.split("_")[0] == subject_id]

    eeg = np.load(os.path.join(eeg_dir, eeg_file[0]))
    label = np.load(os.path.join(label_dir, label_file[0])).squeeze()

    print(f"[INFO] Subject {subject_id} loaded: EEG={eeg.shape}, Label={label.shape}")
    return eeg, label


def main(config):
    data_path = config["data_path"]
    out_folder = config["out_folder"]
    random_folder = time.strftime("%Y-%m-%d--%H-%M", time.localtime())

    lr = config["lr"]
    n_splits = 5

    for subject_id in range(1, 24):
        print("\n==============================")
        print(f" Subject {subject_id} | 5-Fold Cross Validation ")
        print("==============================")

        eeg_data, labels = load_SEEDVIG_data(data_path, subject_id)

        out_path = os.path.join(
            out_folder, config["network"], f"sub{subject_id}", random_folder
        )
        os.makedirs(out_path, exist_ok=True)
        dictToYaml(os.path.join(out_path, "config.yaml"), config)

        setRandom(config["random_seed"])

        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=config["random_seed"]
        )

        for fold, (train_index, test_index) in enumerate(
            tqdm(
                skf.split(eeg_data, labels),
                total=n_splits,
                desc=f"Subject {subject_id} CV",
            )
        ):
            X_train, X_test = eeg_data[train_index], eeg_data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            train_dataset = eegDataset(X_train, y_train)
            test_dataset = eegDataset(X_test, y_test)

            net_args = config["network_args"]
            net = eval(config["network"])(**net_args)

            loss_func = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=lr)

            fold_path = os.path.join(out_path, f"fold{fold + 1}")
            os.makedirs(fold_path, exist_ok=True)

            model = baseModel(
                net, config, optimizer, loss_func, result_savepath=fold_path
            )
            model.train_test(train_dataset, test_dataset)


if __name__ == "__main__":
    configFile = "config/csanet_vig.yaml"
    with open(configFile, "r", encoding="utf-8") as f:
        config = yaml.full_load(f)
    main(config)

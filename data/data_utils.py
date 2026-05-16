import os
import random

import numpy as np


def load_BCI42_data(dataset_path, data_file):
    data_path = os.path.join(dataset_path, data_file + "_data.npy")
    label_path = os.path.join(dataset_path, data_file + "_label.npy")

    data = np.load(data_path)
    label = np.load(label_path).squeeze() - 1

    print(data_file, "load success")

    # Shuffle
    data, label = shuffle_data(data, label)

    print("Data shape: ", data.shape)
    print("Label shape: ", label.shape)

    return data, label


def load_HGD_data(dataset_path, data_file, label_file):
    data = []
    label = []
    data_path = os.path.join(dataset_path, data_file)
    label_path = os.path.join(dataset_path, label_file)

    data = np.load(data_path)
    label = np.load(label_path).squeeze()

    print(data_file, "load success")

    # Shuffle
    data, label = shuffle_data(data, label)

    print("Data shape: ", data.shape)
    print("Label shape: ", label.shape)

    return data, label


def load_SEED_5_fold(root, nSub, fold):
    all_data = np.load(root + "S%d_session1.npy" % nSub, allow_pickle=True)
    all_label = np.load(root + "S%d_session1_label.npy" % nSub, allow_pickle=True)

    train_data = []
    train_label = []
    test_data = []
    test_label = []

    for tri in range(np.shape(all_data)[0]):
        tmp_tri = np.array(all_data[tri])  # shape: (T, C)
        tmp_tri_label = np.array(all_label[tri])

        one_fold_num = np.shape(tmp_tri)[0] // 5
        tri_num = one_fold_num * 5
        tmp_tri_idx = np.arange(tri_num)

        test_idx = np.arange(one_fold_num * fold, one_fold_num * (fold + 1))
        train_idx = np.delete(tmp_tri_idx, test_idx)

        train_seg = tmp_tri[train_idx]
        test_seg = tmp_tri[test_idx]

        if np.isscalar(tmp_tri_label) or tmp_tri_label.ndim == 0:
            train_lab = np.full(len(train_idx), tmp_tri_label)
            test_lab = np.full(len(test_idx), tmp_tri_label)
        else:
            train_lab = tmp_tri_label[train_idx]
            test_lab = tmp_tri_label[test_idx]

        train_data.append(train_seg)
        train_label.append(train_lab)
        test_data.append(test_seg)
        test_label.append(test_lab)

    train_data = np.concatenate(train_data, axis=0)  # (N_train, C)
    test_data = np.concatenate(test_data, axis=0)  # (N_test, C)

    train_label = np.concatenate(train_label, axis=0)  # (N_train,)
    test_label = np.concatenate(test_label, axis=0)  # (N_test,)

    train_data = np.expand_dims(train_data, axis=1)
    test_data = np.expand_dims(test_data, axis=1)

    # shuffle
    shuffle_idx = np.random.permutation(len(train_data))
    train_data = train_data[shuffle_idx]
    train_label = train_label[shuffle_idx] + 1
    test_label = test_label + 1

    # z-score
    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    return train_data, train_label, test_data, test_label


def shuffle_data(data, label):
    index = [i for i in range(len(data))]
    random.shuffle(index)
    shuffle_data = data[index]
    shuffle_label = label[index]
    return shuffle_data, shuffle_label

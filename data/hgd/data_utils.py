import numpy as np
import random
import scipy.signal as signal
import scipy.io as io
import os
import resampy
from scipy.signal import butter, filtfilt


# def load_BCI42_data(dataset_path, data_file):
#     data_path = os.path.join(dataset_path, data_file + '_data.npy')
#     label_path = os.path.join(dataset_path, data_file + '_label.npy')

#     data = np.load(data_path)
#     label = np.load(label_path).squeeze()-1

#     print(data_file, 'load success')

#     #Shuffle
#     data, label = shuffle_data(data, label)

#     print('Data shape: ', data.shape)
#     print('Label shape: ', label.shape)

#     return data, label

def preprocess_filt(data, low_cut=0.1, high_cut=40, fs=500, order=4):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    # 确保 padlen 是基于时间维度（最后一个）
    padlen = data.shape[-1] // 3
    # 指定滤波轴为 -1（即最后一个维度）
    proced = filtfilt(b, a, data, axis=-1, padlen=padlen)
    return proced

def load_BCI42_data_train(dataset_path, data_file):
    data_path = os.path.join("/home/cxr/workspace/数据处理/邓憋尿训练train_data.npy")
    label_path = os.path.join("/home/cxr/workspace/数据处理/邓憋尿训练train_label.npy")

    data = np.load(data_path)
    print(data)
    label = np.load(label_path).squeeze()
    print(label)

    print(data_file, 'load success')

    for i in range(60):
        data[i] = preprocess_filt(data[i])

    #Shuffle
    data, label = shuffle_data(data, label)

    print('Data shape: ', data.shape)
    print('Label shape: ', label.shape)

    return data, label

def load_BCI42_data_test(dataset_path, data_file):
    data_path = os.path.join("/home/cxr/workspace/数据处理/邓憋尿测试test_data.npy")
    label_path = os.path.join("/home/cxr/workspace/数据处理/邓憋尿测试test_label.npy")

    data = np.load(data_path)
    label = np.load(label_path).squeeze()

    print(data_file, 'load success')

    for i in range(20):
        data[i] = preprocess_filt(data[i])

    #Shuffle
    data, label = shuffle_data(data, label)

    print('Data shape: ', data.shape)
    print('Label shape: ', label.shape)

    return data, label

def load_SEED_data(dataset_path, data_file):
    data_path = os.path.join(dataset_path, data_file + '_data.npy')
    label_path = os.path.join(dataset_path, data_file + '_labels.npy')

    data = np.load(data_path)
    segment = len(data)//3
    data = data[segment:segment*2, :, :]

    label = np.load(label_path) + 1
    label = label[segment:segment*2,]

    print(data_file, 'load success')

    #Shuffle
    data, label = shuffle_data(data, label)

    print('Data shape: ', data.shape)
    print('Label shape: ', label.shape)

    return data, label


def load_SEED_5_fold(dataset_path, data_file):
    data_path = os.path.join(dataset_path, data_file + '_data.npy')
    label_path = os.path.join(dataset_path, data_file + '_labels.npy')

    data = np.load(data_path)
    segment = len(data)//3
    data = data[segment:segment*2, :, :]

    label = np.load(label_path) + 1
    label = label[segment:segment*2,]

    print(data_file, 'load success')

    #Shuffle
    data, label = shuffle_data(data, label)

    print('Data shape: ', data.shape)
    print('Label shape: ', label.shape)

    return data, label

# import numpy as np
#
# def load_SEED_5_fold(root, nSub, fold):
#     # 加载数据和标签
#     all_data = np.load(root + 'S%d_session1.npy' % nSub, allow_pickle=True)
#     all_label = np.load(root + 'S%d_session1_label.npy' % nSub, allow_pickle=True)
#
#     train_data = []
#     train_label = []
#     test_data = []
#     test_label = []
#
#     for tri in range(np.shape(all_data)[0]):
#         tmp_tri = np.array(all_data[tri])          # shape: (T, C)
#         tmp_tri_label = np.array(all_label[tri])   # 可能是标量或数组
#
#         one_fold_num = np.shape(tmp_tri)[0] // 5
#         tri_num = one_fold_num * 5
#         tmp_tri_idx = np.arange(tri_num)
#
#         test_idx = np.arange(one_fold_num * fold, one_fold_num * (fold + 1))
#         train_idx = np.delete(tmp_tri_idx, test_idx)
#
#         # 提取训练/测试片段
#         train_seg = tmp_tri[train_idx]
#         test_seg = tmp_tri[test_idx]
#
#         # 处理标签：确保与数据长度对齐
#         if np.isscalar(tmp_tri_label) or tmp_tri_label.ndim == 0:
#             # 如果是标量标签，扩展为与片段等长的数组
#             train_lab = np.full(len(train_idx), tmp_tri_label)
#             test_lab = np.full(len(test_idx), tmp_tri_label)
#         else:
#             # 如果标签本身是时间对齐的（如每个时间点有标签）
#             train_lab = tmp_tri_label[train_idx]
#             test_lab = tmp_tri_label[test_idx]
#
#         train_data.append(train_seg)
#         train_label.append(train_lab)
#         test_data.append(test_seg)
#         test_label.append(test_lab)
#
#     # 合并所有 trial 的片段
#     train_data = np.concatenate(train_data, axis=0)      # (N_train, C)
#     test_data = np.concatenate(test_data, axis=0)        # (N_test, C)
#
#     train_label = np.concatenate(train_label, axis=0)    # (N_train,)
#     test_label = np.concatenate(test_label, axis=0)      # (N_test,)
#
#     # 添加通道维度（例如用于 CNN）：(N, 1, C)
#     train_data = np.expand_dims(train_data, axis=1)
#     test_data = np.expand_dims(test_data, axis=1)
#
#     # 打乱训练集
#     shuffle_idx = np.random.permutation(len(train_data))
#     train_data = train_data[shuffle_idx]
#     train_label = train_label[shuffle_idx] + 1
#     test_label = test_label + 1
#
#     # 标准化：仅用训练集统计量
#     mean = np.mean(train_data)
#     std = np.std(train_data)
#     train_data = (train_data - mean) / std
#     test_data = (test_data - mean) / std
#
#     return train_data, train_label, test_data, test_label
#
# # train_data, train_label, test_data, test_label = load_SEED_5_fold("/mnt/", 1, 0)
# # print(train_data.shape)
# # print(train_label.shape)
# # print(test_data.shape)
# # print(test_label.shape)

def load_HGD_data(dataset_path, data_file, label_file):
    data = []
    label = []
    data_path = os.path.join(dataset_path, data_file)
    label_path = os.path.join(dataset_path, label_file)

    data = np.load(data_path)
    label = np.load(label_path).squeeze()

    print(data_file, 'load success')

    #Shuffle
    data, label = shuffle_data(data, label)

    print('Data shape: ', data.shape)
    print('Label shape: ', label.shape)

    return data, label

def shuffle_data(data, label):
    index = [i for i in range(len(data))]
    random.shuffle(index)
    shuffle_data = data[index]
    shuffle_label = label[index]
    return shuffle_data, shuffle_label
from random import randint

import numpy


def data_augmentation(data, label, index, value_mine, value_max, aug_size):
    """
    对数据进行数据增强，生成保护属性不同，非保护属性，标签相同的样本
    :return:
    """
    aug = []
    for i in range(label.shape[0]):
        # d = data[i].tolist()
        data_size = 0
        data_list = []
        data_existed = [data[i].tolist()]
        while data_size < aug_size:
            aug_data = data[i].tolist()
            while exist_data(data_existed, aug_data, index):
                for j in range(len(index)):
                    new = (randint(value_mine[j], value_max[j]) - value_mine[j]) / (value_max[j] - value_mine[j])
                    aug_data[index[j]] = new
            data_existed.append(aug_data)
            data_size += 1
            data_list.append(aug_data + label[i].tolist())
        aug.append(data_list)
    return numpy.array(aug)


def data_augmentation_all(data, label, index, value_mine, value_max):
    """
    对数据进行数据增强，生成保护属性不同，非保护属性，标签相同的样本
    :return:
    """
    aug = []
    for i in range(label.shape[0]):
        data_list = []
        for a_0 in range(value_max[0] - value_mine[0]):
            for a_1 in range(value_max[1] - value_mine[1]+1):
                for a_2 in range(value_max[2] - value_mine[2]+1):
                    aug_data = data[i].tolist()
                    aug_data[index[0]] = a_0/(value_max[0] - value_mine[0])
                    aug_data[index[1]] = a_1/(value_max[1] - value_mine[1])
                    aug_data[index[2]] = a_2/(value_max[2] - value_mine[2])
                    data_list.append(aug_data + label[i].tolist())
        aug.append(data_list)
    return numpy.array(aug)


def exist_data(data_list, data, index):
    """
    checking whether data exist in the data list
    :return:
    """
    for d in data_list:
        exist_cond = []
        for i in range(len(index)):
            if d[index[i]] == data[index[i]]:
                exist_cond.append(True)
            else:
                exist_cond.append(False)
        if False in exist_cond:
            pass
        else:
            return True
    return False


def data_augmentation_test(data, label, index, value_mine, value_max):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本
    :return:
    """
    aug = []
    for i in range(label.shape[0]):
        aug_data_max = data[i].tolist()
        aug_data_min = data[i].tolist()
        for j in range(len(index)):
            aug_data_max[index[j]] = (value_max[j] - value_mine[j]) / (value_max[j] - value_mine[j])
            aug_data_min[index[j]] = (value_mine[j] - value_mine[j]) / (value_max[j] - value_mine[j])
        data_list = [aug_data_max + label[i].tolist(), aug_data_min + label[i].tolist()]
        aug.append(data_list)
    return numpy.array(aug)

import numpy
import random

import pandas

from utils.util_result import get_numerical_data_max_min, get_data_compas, write_list_2D, normalization
from utils.utils_data_augmentation import data_augmentation, data_augmentation_test


def get_values(list_array):
    """
    获取list的所有可能取值
    :param list_array:
    :return:
    """
    result = []
    for i in range(len(list_array)):
        if list_array[i] not in result:
            result.append(list_array[i])
    return result


def numerical_compas():
    """
    读取compas数据集，将其保存为text格式
    :return:
    """
    compas_file = "compas_data/compas-scores-two-years.csv"
    df = pandas.read_csv(compas_file)
    df = df[['sex', 'age', 'race', 'decile_score', 'priors_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out',
             'c_charge_degree', 'is_recid', 'is_violent_recid', 'v_decile_score', 'two_year_recid']]

    # Indices of data samples to keep
    ix = df['days_b_screening_arrest'] <= 30
    ix = (df['days_b_screening_arrest'] >= -30) & ix
    ix = (df['is_recid'] != -1) & ix
    ix = (df['c_charge_degree'] != "O") & ix
    # ix = (df['score_text'] != 'N/A') & ix
    df = df.loc[ix, :]
    df['length_of_stay'] = (pandas.to_datetime(df['c_jail_out']) - pandas.to_datetime(df['c_jail_in'])).apply(
        lambda x: x.days)
    # 保存category 的所有取值
    data = df[['sex', 'race', 'c_charge_degree']].values
    sex = data[:, 0].reshape(-1)
    race = data[:, 1].reshape(-1)
    c_charge_degree = data[:, 2].reshape(-1)
    sex_values = get_values(list_array=sex)
    race_values = get_values(list_array=race)
    c_charge_degree_values = get_values(list_array=c_charge_degree)
    attribute_file = "compas_data/attribute_information.txt"
    attr_inf = [sex_values, race_values, c_charge_degree_values]
    write_list_2D(file_name=attribute_file, data=attr_inf)

    # 进行数据转换
    def transform_data(x, d):
        return d.index(x)

    df['sex'] = df['sex'].apply(lambda x: transform_data(x, sex_values))
    df['race'] = df['race'].apply(lambda x: transform_data(x, race_values))
    df['c_charge_degree'] = df['c_charge_degree'].apply(lambda x: transform_data(x, c_charge_degree_values))

    data = df[['age', 'race', 'sex', 'decile_score', 'priors_count', 'days_b_screening_arrest', 'length_of_stay',
               'c_charge_degree', 'is_recid', 'is_violent_recid', 'v_decile_score', 'two_year_recid']].values

    transf_file = "compas_data/transform_data.txt"
    write_list_2D(file_name=transf_file, data=data)


def normalization_compas(input_file, max_file, output_file):
    """
    对数据进行预处理，qualitative属性进行one hot encode 编码，numerical连续属性进行归一化
    :return:
    """
    max_file = open(max_file, 'r')
    max_data = []
    min_data = []
    for data in max_file:
        max_min = data.strip().split(',')
        max_data.append(max_min[0])
        min_data.append(max_min[1])
    max_file.close()
    input_file = open(input_file, 'r')
    output_file = open(output_file, 'w')
    for line in input_file:
        keys = line.strip().split(',')
        data = []
        for i in range(11):
            # numerical属性进行归一化
            data.append("{:.4f}".format(normalization(data=keys[i], max_size=max_data[i], min_size=min_data[i])))
        data.append(keys[11])
        print(','.join(data), end="\n", file=output_file)
    input_file.close()
    output_file.close()


def split_train_test(input_f, train_f, test_f, rate):
    """
    对credit数据集划分训练集，测试集
    :return:
    """
    input_file = open(input_f, "r")
    train_file = open(train_f, "w")
    test_file = open(test_f, "w")
    for line in input_file:
        i = random.random()
        if i > rate:
            print(line.strip(), end='\n', file=train_file)
        else:
            print(line.strip(), end='\n', file=test_file)
    train_file.close()
    test_file.close()


if __name__ == "__main__":
    # "转换compas为numerical数据"
    # numerical_compas()
    #
    # "获取数据集中 numerical 中的最大值最小值"
    # transform_file = "compas_data/transform_data.txt"
    # max_mine_file = 'compas_data/max_mine_data.txt'
    # get_numerical_data_max_min(input_file=transform_file, output_file=max_mine_file, length=12, width=2)
    #
    # "对原始数据集划分训练集，测试集"
    # train_file = "compas_data/0.8_train.txt"
    # test_file = "compas_data/0.2_test.txt"
    # rate = 0.2
    # split_train_test(input_f=transform_file, train_f=train_file, test_f=test_file, rate=rate)
    #
    # "对训练集，测试集进行数据编码，归一化"
    # "train dataset"
    # norm_train = "compas_data/train.txt"
    # normalization_compas(input_file=train_file, max_file=max_mine_file, output_file=norm_train)
    # "test dataset"
    # norm_test = "compas_data/test.txt"
    # normalization_compas(input_file=test_file, max_file=max_mine_file, output_file=norm_test)

    norm_train = "compas_data/train.txt"
    norm_test = "compas_data/test.txt"

    "生成sex的训练及测试数据"
    x, y = get_data_compas(filename=norm_train)
    aug = data_augmentation(x, y, [2], [0], [1], 1)
    numpy.save("compas_data/sex_train.npz", aug)

    x, y = get_data_compas(filename=norm_test)
    aug = data_augmentation_test(x, y, [2], [0], [1])
    numpy.save("compas_data/sex_test.npz", aug)

    "生成race的训练及测试数据"
    x, y = get_data_compas(filename=norm_train)
    aug = data_augmentation(x, y, [1], [0], [5], 5)
    numpy.save("compas_data/race_train.npz", aug)

    x, y = get_data_compas(filename=norm_test)
    aug = data_augmentation_test(x, y, [1], [0], [5])
    numpy.save("compas_data/race_test.npz", aug)

    "生成age的训练及测试数据"
    x, y = get_data_compas(filename=norm_train)
    aug = data_augmentation(x, y, [0], [18], [96], 70)
    numpy.save("compas_data/age_train.npz", aug)

    x, y = get_data_compas(filename=norm_test)
    aug = data_augmentation_test(x, y, [0], [18], [96])
    numpy.save("compas_data/age_test.npz", aug)

    "生成age,race,sex的训练及测试数据"
    x, y = get_data_compas(filename=norm_train)
    aug = data_augmentation(x, y, [0, 1, 2], [18, 0, 0], [96, 5, 1], 70)
    numpy.save("compas_data/multiple_train.npz", aug)

    x, y = get_data_compas(filename=norm_test)
    aug = data_augmentation_test(x, y, [0, 1, 2], [18, 0, 0], [96, 5, 1])
    numpy.save("compas_data/multiple_test.npz", aug)

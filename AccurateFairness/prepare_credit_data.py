import numpy

from prepare_compas_data import split_train_test
from utils.util_result import get_numerical_data_max_min, normalization, get_data_credit
from utils.utils_data_augmentation import data_augmentation, data_augmentation_test


def numerical_credit(input_f, output_f):
    """
    将credit数据转换为numerical数据
    male 1
    female 0
    :return:
    """
    input_1 = open(input_f, 'r')
    output_1 = open(output_f, 'w')
    for l in input_1:
        # 解析数据，将原始数据转换为numerical
        raw_data = l.split()
        p_data = []
        n_data = []
        label = []
        for i in range(21):
            if raw_data[i].find("A") > -1:
                if i == 8:
                    # 解析性别，婚姻状态
                    d = raw_data[i].replace("A{}".format(i + 1), "").strip()
                    if d == "1":
                        p_data.append("1")
                        n_data.append("0")
                    elif d == "2":
                        p_data.append("0")
                        n_data.append("0")
                    elif d == "3":
                        p_data.append("1")
                        n_data.append("1")
                    elif d == "4":
                        p_data.append("1")
                        n_data.append("2")
                    elif d == "5":
                        p_data.append("0")
                        n_data.append("1")
                else:
                    d = raw_data[i].replace("A{}".format(i + 1), "").strip()
                    n_data.append(d)
            else:
                # 保护属性年龄
                if i == 12:
                    p_data.append(raw_data[i].strip())
                # 标签
                elif i == 20:
                    label.append(raw_data[i].strip())
                else:
                    n_data.append(raw_data[i].strip())
        numerical_data = n_data + p_data + label
        line = ','.join(numerical_data) + "\n"
        output_1.write(line)
    input_1.close()
    output_1.close()


def normalization_credit(input_file, max_file, output_file):
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
        label = []
        for i in range(21):
            data.append("{:.4f}".format(normalization(data=keys[i], max_size=max_data[i], min_size=min_data[i])))

        # label: 1 = Good,  2 = Bad
        if float(keys[21]) == 2:
            label.append("0")
        else:
            label.append("1")
        d = data + label
        print(','.join(d), end="\n", file=output_file)
    input_file.close()
    output_file.close()


if __name__ == "__main__":
    # "将字符数据转换为numerical数据"
    # raw_file = 'credit_data/german.data'
    # numerical_file = 'credit_data/transform_data.txt'
    # numerical_credit(input_f=raw_file, output_f=numerical_file)
    #
    # "获取数据集中 numerical complete中的最大值最小值"
    # max_mine_file = 'credit_data/max_mine_data.txt'
    # get_numerical_data_max_min(input_file=numerical_file, output_file=max_mine_file, length=22, width=2)
    #
    # "对原始数据集划分训练集，测试集"
    # sp_train_file = "credit_data/split_train.txt"
    # sp_test_file = "credit_data/split_test.txt"
    # rate = 0.2
    # split_train_test(input_f=numerical_file, train_f=sp_train_file, test_f=sp_test_file, rate=rate)
    #
    # " 对训练集，测试集进行数据编码，归一化 "
    # "train dataset"
    # train_file = "credit_data/train.txt"
    # normalization_credit(input_file=sp_train_file, max_file=max_mine_file, output_file=train_file)
    # "test dataset"
    # test_file = "credit_data/test.txt"
    # normalization_credit(input_file=sp_test_file, max_file=max_mine_file, output_file=test_file)

    train_file = "credit_data/train.txt"
    test_file = "credit_data/test.txt"

    "生成sex的训练及测试数据"
    x, y = get_data_credit(filename=train_file)
    aug = data_augmentation(x, y, [19], [0], [1], 1)
    numpy.save("credit_data/sex_train.npz", aug)

    x, y = get_data_credit(filename=test_file)
    aug = data_augmentation_test(x, y, [19], [0], [1])
    numpy.save("credit_data/sex_test.npz", aug)

    "生成age的训练及测试数据"
    x, y = get_data_credit(filename=train_file)
    aug = data_augmentation(x, y, [20], [19], [75], 50)
    numpy.save("credit_data/age_train.npz", aug)

    x, y = get_data_credit(filename=test_file)
    aug = data_augmentation_test(x, y, [20], [19], [75])
    numpy.save("credit_data/age_test.npz", aug)

    "生成sex, age的训练及测试数据"
    x, y = get_data_credit(filename=train_file)
    aug = data_augmentation(x, y, [19, 20], [0, 19], [1, 75], 70)
    numpy.save("credit_data/multiple_train.npz", aug)

    x, y = get_data_credit(filename=test_file)
    aug = data_augmentation_test(x, y, [19, 20], [0, 19], [1, 75])
    numpy.save("credit_data/multiple_test.npz", aug)

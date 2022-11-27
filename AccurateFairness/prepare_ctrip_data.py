import numpy
from utils.util_result import write_list_1D, get_data_hotel
from utils.utils_data_augmentation import data_augmentation_test


def preprocessing_test_hotel_data(hotel_data):
    """
    对test文件中的数据进行处理选择酒店信息，房价信息
    :return:
    """
    test_data_index = [2, 3, 4, 5, 6, 9]
    data = []
    for i in range(len(hotel_data)):
        if i in test_data_index:
            if i == 2:
                data.append(hotel_data[i].split("-")[-1])
            elif i == 6:
                data.append(hotel_data[i])
            elif i == 9:
                data.append(hotel_data[i])
            else:
                data.append(hotel_data[i].split("_")[1])
    return data


def check_label(labels):
    """
    检查label的合法性
    :return:
    """
    if float(labels[0]) < 2 and float(labels[1]) < 2 and float(labels[2]) < 4 and \
            float(labels[3]) < 5 and float(labels[4]) < 2 and float(labels[5]) < 3 and \
            float(labels[6]) < 2 and float(labels[7]) < 6:
        return True
    else:
        return False


def preprocessing_test_file():
    """
    对测试数据进行预处理
    :return:
    """
    output_file = "ctrip_dataa/test.txt"
    output_file = open(output_file, "w")
    hotel_test = "ctrip_data/competition_test.txt"
    with open(hotel_test, "r") as f:
        i = 0
        j = 0
        for l in f:
            i = i + 1
            if i == 1:
                continue
            value = l.strip().split()[:48]
            if "NULL" not in value:
                hotel_data = preprocessing_test_hotel_data(value[0:12])
                label = value[12:20]
                user_data = value[26:32]
                if check_label(label):
                    data = hotel_data + user_data + label
                    j = j + 1
                    write_list_1D(file=output_file, data=data)
    output_file.close()


def preprocessing_train_hotel_data(hotel_data):
    """
    对test文件中的数据进行处理选择酒店信息，房价信息
    :return:
    """
    test_data_index = [2, 3, 4, 5, 7, 10]
    data = []
    for i in range(len(hotel_data)):
        if i in test_data_index:
            if i == 2:
                data.append(hotel_data[i].split("-")[-1])
            elif i == 7:
                data.append(hotel_data[i])
            elif i == 10:
                data.append(hotel_data[i])
            else:
                data.append(hotel_data[i].split("_")[1])
    return data


def preprocessing_train_file():
    """
    对训练数据进行预处理
    :return:
    """
    output_file = "ctrip_data/train.txt"
    output_file = open(output_file, "w")
    hotel_test = "ctrip_data/competition_train.txt"
    with open(hotel_test, "r") as f:
        i = 0
        j = 0
        for l in f:
            i = i + 1
            if i == 1:
                continue
            value = l.strip().split()[:49]
            if "NULL" not in value:
                hotel_data = preprocessing_train_hotel_data(value[0:13])
                label = value[13:21]
                user_data = value[27:33]
                if check_label(label):
                    data = hotel_data + user_data + label
                    j = j + 1
                    write_list_1D(file=output_file, data=data)
    print(i)
    print(j)
    output_file.close()


def get_data_max_min(input_file, output_file, length, width):
    """
    get the max and min feature value in the file
    :param input_file:
    :param output_file:
    :param length:
    :param width:
    :return:
    """
    input_file = open(input_file, "r")
    output_file = open(output_file, "w")
    max_min = numpy.ones(shape=(length, width), dtype=int)
    initialization = input_file.readline()
    initialization_keys = initialization.strip().split(',')
    for k in range(length):
        if 0 < k < 4:
            max_min[k, 0] = float(initialization_keys[k])
            max_min[k, 1] = float(initialization_keys[k])
        else:
            max_min[k, 0] = float(initialization_keys[k])
            max_min[k, 1] = float(initialization_keys[k])
    for line in input_file:
        keys = line.split(',')
        for k in range(length):
            if 0 < k < 4:
                if float(keys[k]) > max_min[k, 0]:
                    max_min[k, 0] = float(keys[k])
                elif float(keys[k]) < max_min[k, 1]:
                    max_min[k, 1] = float(keys[k])
            else:
                if float(keys[k]) >= max_min[k, 0]:
                    max_min[k, 0] = float(keys[k])
                elif float(keys[k]) < max_min[k, 1]:
                    max_min[k, 1] = float(keys[k])
    input_file.close()
    for i in range(length):
        print(str(max_min[i, 0]) + "," + str(max_min[i, 1]), end='\n', file=output_file)
    output_file.close()


def get_hotel_max_min(file):
    """
    获取携程数据集中各属性的最大最小值
    :return:
    """
    max_data = []
    min_data = []
    with open(file, "r") as f:
        for l in f:
            data = l.strip().split(",")
            max_data.append(float(data[0]))
            min_data.append(float(data[1]))
    f.close()
    return max_data, min_data


def encode_feature(max_file, data_file, output_file):
    """
    获取最大值最小值
    :return:
    """
    max_file = open(max_file, 'r')
    max_data = []
    min_data = []
    for data in max_file:
        max_min = data.strip().split(',')
        max_data.append(int(max_min[0]))
        min_data.append(int(max_min[1]))
    max_file.close()
    out = open(output_file, "w")
    with open(data_file, "r") as d_file:
        for l in d_file:
            if l.find("start") > -1 or l.find("end") > -1:
                continue
            d = l.strip().split(",")
            result = []
            for i in range(len(d)):
                if i < 12:
                    result.append("{:.4f}".format((float(d[i]) - min_data[i]) / (max_data[i] - min_data[i])))
                else:
                    if i == 16:
                        continue
                    else:
                        result.append(d[i])
            write_list_1D(file=out, data=result)
        d_file.close()
    out.close()


if __name__ == "__main__":
    # "对原始数据集进行属性选择，以及转换为numerical"
    # preprocessing_test_file()
    # preprocessing_train_file()
    # "获取每个属性的最大最小值，用于编码"
    # train_data_file = "ctrip_data/train.txt"
    # train_max_min_file = "ctrip_data/train_max_min.txt"
    # get_data_max_min(input_file=train_data_file, output_file=train_max_min_file, length=20, width=2)
    # test_data_file = "ctrip_data/test.txt"
    # test_max_min_file = "ctrip_data/test_max_min.txt"
    # get_data_max_min(input_file=test_data_file, output_file=test_max_min_file, length=20, width=2)
    # "对属性进行编码预处理"
    # train_encode_file = "ctrip_data/encode_train.txt"
    # encode_feature(train_max_min_file, train_data_file, train_encode_file)
    # test_encode_file = "ctrip_data/encode_test.txt"
    # encode_feature(test_max_min_file, test_data_file, test_encode_file)

    "生成训练及测试数据"
    encode_file = "ctrip_data/encode_train.txt"
    x, y = get_data_hotel(filename=encode_file)
    aug = data_augmentation_test(x, y, [6, 7, 8, 9, 10, 11], [-117445, -37, 1, 1, 3, 302], [0, 250, 11, 13, 21, 308])
    numpy.save("ctrip_data/multiple_train.npz", aug)

    encode_file = "ctrip_data/encode_test.txt"
    x, y = get_data_hotel(filename=encode_file)
    aug = data_augmentation_test(x, y, [6, 7, 8, 9, 10, 11], [-86274, -29, 1, 1, 3, 302], [0, 201, 11, 13, 23, 308])
    numpy.save("ctrip_data/multiple_test.npz", aug)

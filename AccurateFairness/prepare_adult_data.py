import numpy
from utils.utils_data_augmentation import data_augmentation, data_augmentation_test
from utils.util_result import get_data_adult, get_numerical_data_max_min
from utils.utils_data_clean_adult import dictionary_adult, numerical_adult, normalization_adult

if __name__ == "__main__":
    # "获取属性dictionary,合并部分属性,根据属性字典将raw data转换为numerical"
    # attr_file = 'adult_data/attribute_information.txt'
    # dic_file = 'adult_data/attribute_dictionary.txt'
    # name, value = dictionary_adult(attribute_file=attr_file, dictionary_file=dic_file)
    #
    # "将原始数据转换 numerical数据"
    # "train"
    # raw_train = 'adult_data/adult.data'
    # transform_train = 'adult_data/numerical_train.txt'
    # clean_train = "adult_data/clean_train.txt"
    # numerical_adult(input_1=raw_train, output_1=transform_train, output_2=clean_train, dictionary=value)
    # "test"
    # raw_test = 'adult_data/adult.test'
    # transform_test = 'adult_data/numerical_test.txt'
    # clean_test = "adult_data/clean_txt.txt"
    # numerical_adult(input_1=raw_test, output_1=transform_test, output_2=clean_test, dictionary=value)
    #
    # "获取数据集中 numerical complete中的最大值最小值"
    # "train"
    # train_max_min = 'adult_data/train_max_mine.txt'
    # get_numerical_data_max_min(input_file=transform_train, output_file=train_max_min, length=14, width=2)
    # "test"
    # test_max_mine = 'adult_data/test_max_mine.txt'
    # get_numerical_data_max_min(input_file=transform_test, output_file=test_max_mine, length=14, width=2)
    #
    # "对进行数据预处理后的数据进行数据编码，归一化"
    # "train dataset"
    # train_file = "adult_data/train.txt"
    # normalization_adult(input_file=transform_train, max_file=train_max_min, output_file=train_file)
    # "test dataset"
    # test_file = "adult_data/test.txt"
    # normalization_adult(input_file=transform_test, max_file=test_max_mine, output_file=test_file)

    train_file = "adult_data/train.txt"
    test_file = "adult_data/test.txt"

    "生成sex的训练及测试数据"
    x, y = get_data_adult(filename=train_file)
    aug = data_augmentation(x, y, [12], [0], [1], 1)
    numpy.save("adult_data/sex_train.npz", aug)

    x, y = get_data_adult(filename=test_file)
    aug = data_augmentation_test(x, y, [12], [0], [1])
    numpy.save("adult_data/sex_test.npz", aug)

    "生成race的训练及测试数据"
    x, y = get_data_adult(filename=train_file)
    aug = data_augmentation(x, y, [11], [0], [4], 4)
    numpy.save("adult_data/race_train.npz", aug)
    x, y = get_data_adult(filename=test_file)
    aug = data_augmentation_test(x, y, [11], [0], [4])
    numpy.save("adult_data/race_test.npz", aug)

    "生成age的训练及测试数据"
    x, y = get_data_adult(filename=train_file)
    aug = data_augmentation(x, y, [10], [17], [90], 70)
    numpy.save("adult_data/age_train.npz", aug)

    aug = data_augmentation_test(x, y, [10], [17], [90])
    numpy.save("adult_data/age_test.npz", aug)

    "生成age,race,sex的训练及测试数据"
    aug_10 = data_augmentation(x, y, [10, 11, 12], [17, 0, 0], [90, 4, 1], 70)
    numpy.save("adult_data/multiple_train_70.npz", aug_10)

    x, y = get_data_adult(filename=test_file)
    aug = data_augmentation_test(x, y, [10, 11, 12], [17, 0, 0], [90, 4, 1])
    numpy.save("adult_data/multiple_test.npz", aug)

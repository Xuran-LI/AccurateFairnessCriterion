import json

from utils.util_result import normalization


def combine_feature_adult(feature):
    """
    combine the feature by definition
    :param feature:
    :return:
    """
    # define combined feature
    Unemployed = ["Without-pay", "Never-worked"]
    SL_gov = ["State-gov", "Local-gov"]
    Self_employed = ["Self-emp-inc", "Self-emp-not-inc"]
    Married = ["Married-AF-spouse", "Married-civ-spouse", "Married-spouse-absent"]
    Not_Married = ["Divorced", "Separated", "Widowed"]
    north_america = ["Canada", "Cuba", "Dominican-Republic", "El-Salvador", "Guatemala", "Haiti", "Honduras", "Jamaica",
                     "Mexico", "Nicaragua", "Outlying-US(Guam-USVI-etc)", "Puerto-Rico", "Trinadad&Tobago",
                     "United-States"]
    asia = ["Cambodia", "China", "Hong", "India", "Iran", "Japan", "Laos", "Philippines", "Taiwan", "Thailand",
            "Vietnam"]
    south_america = ["Columbia", "Ecuador", "Peru"]
    europe = ["England", "France", "Germany", "Greece", "Holand-Netherlands", "Hungary", "Ireland", "Italy", "Poland",
              "Portugal", "Scotland", "Yugoslavia"]
    if feature in Unemployed:
        return 'Unemployed'
    elif feature in SL_gov:
        return 'SL_gov'
    elif feature in Self_employed:
        return 'Self_employed'
    elif feature in Married:
        return 'Married'
    elif feature in Not_Married:
        return 'Not_Married'
    elif feature in north_america:
        return 'north_america'
    elif feature in asia:
        return 'asia'
    elif feature in south_america:
        return 'south_america'
    elif feature in europe:
        return 'europe'
    else:
        return feature


def dictionary_adult(attribute_file, dictionary_file):
    """
    create new dictionary which the feature in defined features has the same values
    :param dictionary_file: new dictionary
    :param attribute_file: old dictionary
    :return:
    """
    attribute_file = open(attribute_file, 'r')
    dictionary_file = open(dictionary_file, 'w')
    feature_name = []
    feature_value = {}
    for line in attribute_file:
        feature = line.split(':')
        feature_name.append(feature[0].strip())
        values = feature[1].replace('.', '').strip().split(',')
        if len(values) > 1:
            j = 0
            for value in values:
                if combine_feature_adult(value.strip()) in feature_value:
                    feature_value[value.strip()] = feature_value[combine_feature_adult(value.strip())]
                else:
                    feature_value[combine_feature_adult(value.strip())] = j
                    feature_value[value.strip()] = j
                    j = j + 1
    dictionary_file.write(json.dumps(feature_value))
    attribute_file.close()
    dictionary_file.close()
    return feature_name, feature_value


def transform_label_adult(label):
    if label.find('>50K') > -1:
        return '1\n'
    else:
        return '0\n'


def numerical_adult(input_1, output_1, output_2, dictionary):
    """
    将原始数据转换为数值型数据，对数值型数据按非保护属性，保护属性，label进行重新排序组织。分离完整数据，残缺数据
    :param input_1:
    :param output_1:
    :param dictionary:
    :return:
    """
    input_1 = open(input_1, 'r')
    output_1 = open(output_1, 'w')
    output_2 = open(output_2, "w")
    for line in input_1:
        if line.find('?') > -1:
            continue
        # 对数据进行转换
        output_2.write(line)
        keys = line.split(',')
        none_protect_feature_c = []
        protect_feature_c = []
        none_protect_feature_d = []
        protect_feature_d = []
        label = []
        for k in range(len(keys)):
            # continue protect feature
            if k == 0:
                protect_feature_c.append(keys[k].strip())
            # delete feature fnlwgt
            elif k == 2:
                pass
            # continue  none protect feature
            elif k == 4 or k == 10 or k == 11 or k == 12:
                none_protect_feature_c.append(keys[k].strip())
            elif k == 14:
                label.append(transform_label_adult(label=keys[k]))
            # category  protect feature
            elif k == 8 or k == 9:
                protect_feature_d.append(str(dictionary[keys[k].strip()]))
            # category none protect feature
            else:
                none_protect_feature_d.append(str(dictionary[keys[k].strip()]))

        data = none_protect_feature_c + none_protect_feature_d + protect_feature_c + protect_feature_d + label
        # output_1.write(','.join(data))
        print(",".join('%.4f' % float(d) for d in data).strip(), end='\n', file=output_1)

    input_1.close()
    output_1.close()
    output_2.close()


def normalization_adult(input_file, max_file, output_file):
    """
    对数据进行预处理，离散属性进行one hot encode 编码，连续属性进行归一化
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
        keys = line.split(',')
        if len(keys) < 14:
            continue
        data = []
        for i in range(14):
            if i < 13:
                data.append(
                    "{:.4f}".format(normalization(data=keys[i], max_size=max_data[i], min_size=min_data[i])))
            elif i == 13:
                data.append(keys[i])
        print(','.join([str(s) for s in data]).strip(), end="\n", file=output_file)
    input_file.close()
    output_file.close()

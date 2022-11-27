import numpy
import pandas
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot


def normalization(data, max_size, min_size):
    molecule = float(data) - float(min_size)
    denominator = float(max_size) - float(min_size)
    result = molecule / denominator
    return round(result, 4)


def get_numerical_data_max_min(input_file, output_file, length, width):
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
        max_min[k, 0] = float(initialization_keys[k])
        max_min[k, 1] = float(initialization_keys[k])
    for line in input_file:
        keys = line.split(',')
        for k in range(length):
            if float(keys[k]) >= max_min[k, 0]:
                max_min[k, 0] = float(keys[k])
            elif float(keys[k]) < max_min[k, 1]:
                max_min[k, 1] = float(keys[k])
    input_file.close()
    for i in range(length):
        print(str(max_min[i, 0]) + "," + str(max_min[i, 1]), end='\n', file=output_file)
    output_file.close()


def get_numerical_data(filename):
    """
    get numerical adult_data from file
    :return:
    """
    raw_data = []
    with open(filename, 'r') as train_data:
        for line in train_data:
            data = line.strip("\n").split(",")
            if len(data) > 0:
                raw_data.append(data)
        train_data.close()
    train_data.close()
    return numpy.array(raw_data).astype(float)


def write_list_1D(file, data):
    """
    输出list/numpy  1D数据
    :return:
    """
    # print(",".join('{}'.format(d) for d in data).strip(), end='\n', file=file)
    print(",".join('%.4f' % float(d) for d in data).strip(), end='\n', file=file)


def write_list_2D(file_name, data):
    """
    write 2D list to file
    :return:
    """
    with open(file_name, "w") as f:
        for i in range(len(data)):
            write_list_1D(file=f, data=data[i])
        f.flush()
    f.close()


def get_data_adult(filename):
    """
    获取adult的numerical数据
    :param filename:
    :return:
    """
    data = get_numerical_data(filename=filename)
    x, y = numpy.split(data, [13, ], axis=1)
    return x, y


def get_data_hotel(filename):
    """
    获取ctrip数据
    :return:
    """
    data = pandas.read_csv(filename, header=None).values
    data, label = numpy.split(data, [12, ], axis=1)
    return data, label


def get_data_compas(filename):
    """
    获取compas的numerical数据
    :return:
    """
    data = get_numerical_data(filename=filename)
    x, y = numpy.split(data, [11, ], axis=1)
    return x, y


def get_data_credit(filename):
    """
    获取credit的numerical数据
    :param filename:
    :return:
    """
    data = get_numerical_data(filename=filename)
    x, y = numpy.split(data, [21, ], axis=1)
    return x, y


def compare_two_numpy(label, predicate):
    """
    基于原始标签，及模型预测计算精度
    :return:
    """
    condition = []
    for i in range(label.shape[0]):
        # print(label[i] == predicate[i])
        if (label[i] == predicate[i]).all():
            condition.append(True)
        else:
            condition.append(False)
    return numpy.array(condition).astype(bool)


def numpy_equal_to_value(input_n, value):
    """
    计算离散属性取值为value的condition
    compute the condition which value in dispersed attribute array are same as value
    :return:
    """
    condition = []
    for i in range(input_n.shape[0]):
        if input_n[i] == value:
            condition.append(True)
        else:
            condition.append(False)
    return numpy.array(condition).astype(bool)


def logical_and_cond(c1, c2):
    """
    计算condition1 和 condition2 的逻辑与
    :return:
    """
    condition = numpy.logical_and(c1, c2)
    return condition


def transform_predication(predicates):
    """
    将模型的输出转换为向量形式
    :param predicates:
    :return:
    """
    predicates = numpy.argmax(predicates, axis=1)
    return predicates


def get_accuracy(label, predicate):
    """
    检查模型精度
    :return:
    """
    acc_cond = compare_two_numpy(label=label, predicate=predicate)
    acc_rate = numpy.sum(acc_cond) / acc_cond.shape[0]
    return acc_rate


def get_sp_diff(binary_feature, predicate):
    """
    获取模型的statistical parity difference
    :return:
    """
    # 保护属性分组
    f_0 = numpy_equal_to_value(binary_feature, 0)
    f_1 = numpy_equal_to_value(binary_feature, 1)
    # 预测分组
    p_1 = numpy_equal_to_value(predicate, 1)
    # result
    f_0_p_1 = logical_and_cond(c1=f_0, c2=p_1)
    f_1_p_1 = logical_and_cond(c1=f_1, c2=p_1)
    sp_0 = numpy.sum(f_0_p_1) / numpy.sum(f_0)
    sp_1 = numpy.sum(f_1_p_1) / numpy.sum(f_1)
    sp_d = abs(sp_0 - sp_1)
    return sp_d


def get_eo_avo_diff(binary_feature, label, predicate):
    """
    获取模型的equal opportunity difference
    :return:
    """
    # 保护属性分组
    f_0 = numpy_equal_to_value(binary_feature, 0)
    f_1 = numpy_equal_to_value(binary_feature, 1)
    # 标签分组
    l_0 = numpy_equal_to_value(label, 0)
    l_1 = numpy_equal_to_value(label, 1)
    # 预测分组
    p_0 = numpy_equal_to_value(predicate, 0)
    p_1 = numpy_equal_to_value(predicate, 1)
    # "TP FP TN FN"
    TP_condition = logical_and_cond(l_1, p_1)
    FP_condition = logical_and_cond(l_0, p_1)
    TN_condition = logical_and_cond(l_0, p_0)
    FN_condition = logical_and_cond(l_1, p_0)

    f_0_tp = logical_and_cond(f_0, TP_condition)
    f_0_fp = logical_and_cond(f_0, FP_condition)
    f_0_tn = logical_and_cond(f_0, TN_condition)
    f_0_fn = logical_and_cond(f_0, FN_condition)

    f_1_tp = logical_and_cond(f_1, TP_condition)
    f_1_fp = logical_and_cond(f_1, FP_condition)
    f_1_tn = logical_and_cond(f_1, TN_condition)
    f_1_fn = logical_and_cond(f_1, FN_condition)

    confusion_cond = [f_1_tp, f_1_fn, f_1_fp, f_1_tn, f_0_tp, f_0_fn, f_0_fp, f_0_tn]
    result = []
    for cond in confusion_cond:
        result.append(numpy.sum(cond))
    TPR_1 = (result[0]) / (result[0] + result[1])
    TPR_2 = (result[4]) / (result[4] + result[5])
    FPR_1 = (result[2]) / (result[2] + result[3])
    FPR_2 = (result[6]) / (result[6] + result[7])

    tp_diff = TPR_1 - TPR_2
    fp_diff = FPR_1 - FPR_2
    avo = (abs(tp_diff) + abs(fp_diff)) / 2
    eod = abs(tp_diff)
    return avo, eod


def get_consistency(features, labels, n_neighbors):
    """
    计算个体knn公平性,各样本周围的k领域样本坐标
    :return:
    """
    X = features
    num_samples = X.shape[0]
    y = labels
    # learn a KNN on the features
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
    _, indices = nbrs.kneighbors(X)
    # compute consistency score
    consistency = 0.0
    for i in range(num_samples):
        consistency += numpy.abs(y[i] - numpy.mean(y[indices[i]]))
    consistency = 1.0 - consistency / num_samples

    return consistency


def get_individual_fairness(org_pred, aug_pre):
    """
    获取模型预测结果的individual fairness情况
    :return:
    """
    fair_cond1 = compare_two_numpy(org_pred, aug_pre[0])
    fair_cond2 = compare_two_numpy(org_pred, aug_pre[1])
    fair_cond = numpy.logical_and(fair_cond1, fair_cond2)
    IF = numpy.sum(fair_cond) / fair_cond.shape[0]
    return IF


def get_accurate_fairness(org_label, org_pred, aug_pre):
    """
    获取model预测结果的准确公平性
    :return:
    """
    fair_cond1 = compare_two_numpy(org_pred, aug_pre[0])
    fair_cond2 = compare_two_numpy(org_pred, aug_pre[1])
    fair_cond = numpy.logical_and(fair_cond1, fair_cond2)
    bias_cond = ~fair_cond
    true_cond = compare_two_numpy(org_label, org_pred)
    false_cond = ~true_cond
    true_fair_cond = logical_and_cond(true_cond, fair_cond)
    true_bias_cond = logical_and_cond(true_cond, bias_cond)
    false_fair_cond = logical_and_cond(false_cond, fair_cond)
    false_bias_cond = logical_and_cond(false_cond, bias_cond)
    sum = org_label.shape[0]
    TFR = numpy.sum(true_fair_cond) / sum
    TBR = numpy.sum(true_bias_cond) / sum
    FFR = numpy.sum(false_fair_cond) / sum
    FBR = numpy.sum(false_bias_cond) / sum
    F_recall = TFR / (TFR + FFR)
    F_precision = TFR / (TFR + TBR)
    F_F1 = (2 * F_recall * F_precision) / (F_recall + F_precision)
    return TFR, TBR, FFR, FBR, F_recall, F_precision, F_F1


def test_model(org_label, org_pre, protect):
    """
    获取模型accuracy, SPD,EOD,AVOD,consistency,IF
    :return:
    """
    SPD = get_sp_diff(protect, org_pre)
    EOD, AVOD = get_eo_avo_diff(protect, org_label, org_pre)
    result = [SPD, EOD, AVOD]
    return ["{:.4f}".format(r) for r in result]


def test_model_acc_fair(org_label, org_pre, aug_pre, x, neighbor):
    """
    get model accuracy fairness result
    TFR, TBR, FFR, FBR, F_recall, F_precision, F_F1
    :return:
    """
    TFR, TBR, FFR, FBR, F_recall, F_precision, F_F1 = get_accurate_fairness(org_label, org_pre, aug_pre)
    ACC = get_accuracy(org_label, org_pre)
    IF = get_individual_fairness(org_pre, aug_pre)
    consistency = get_consistency(x, org_pre.reshape(-1, 1), neighbor)[0]
    result = [ACC, consistency, IF, TFR, TBR, FFR, FBR, F_recall, F_precision, F_F1]
    return ["{:.4f}".format(r) for r in result]


def test_model_similar(org_label, aug_pre, aug_protect):
    """
    获取模型similar dataset: SPD,EOD,AVOD
    :return:
    """
    labels = numpy.concatenate((org_label, org_label), axis=0)
    predications = numpy.concatenate((aug_pre[0], aug_pre[1]), axis=0)
    binary_attributes = numpy.concatenate((aug_protect[0], aug_protect[1]), axis=0)
    SPD = get_sp_diff(binary_attributes, predications)
    EOD, AVOD = get_eo_avo_diff(binary_attributes, labels, predications)
    result = [SPD, EOD, AVOD]
    return ["{:.4f}".format(r) for r in result]


def draw_pie_chart(TFR, TBR, FFR, FBR, name):
    labels = 'TFR', 'TBR', 'FBR', 'FFR'
    colors = ["c", "m", "y", "springgreen"]
    fig1, ax1 = pyplot.subplots(1, figsize=(2, 2))
    sizes = [TFR, TBR, FFR, FBR]
    ax1.pie(sizes, autopct='%1.1f%%', colors=colors, shadow=True, startangle=90, normalize=True)
    ax1.axis('equal')
    pyplot.savefig('pic/{}.pdf'.format(name), bbox_inches='tight')
    pyplot.show()


def draw_Fairea_chart(IF, ACC, name):
    methods = ['BL', "SF"]

    fig, ax = pyplot.subplots(figsize=(6, 2))
    for i in range(len(methods)):
        ax.plot(IF[i], ACC[i], "+", color='navy', markersize=8)
        ax.annotate(methods[i], (IF[i] + 0.0015, ACC[i]), fontsize=10)
    pyplot.xlim((0.5, 1.0))
    pyplot.ylim((0.5, 1.0))
    pyplot.xlabel("FTA")
    pyplot.ylabel("ACC")

    rect = pyplot.Rectangle((0, 0), IF[0], ACC[0], facecolor='indianred', label='lose-lose')
    ax.add_patch(rect)
    rect = pyplot.Rectangle((IF[0], 0), 1 - IF[0], ACC[0], facecolor='greenyellow', label='better ACC')
    ax.add_patch(rect)
    rect = pyplot.Rectangle((0, ACC[0]), IF[0], 1 - ACC[0], facecolor='royalblue', label='better FTA')
    ax.add_patch(rect)
    rect = pyplot.Rectangle((IF[0], ACC[0]), 1 - IF[0], 1 - ACC[0], facecolor='goldenrod', label='win-win')
    ax.add_patch(rect)
    pyplot.savefig('pic/{}.pdf'.format(name), bbox_inches='tight')
    pyplot.show()


def get_average(data_list, data_index):
    result = 0
    for i in range(len(data_list)):
        result += float(data_list[i][data_index])

    return result / len(data_list)

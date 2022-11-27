import numpy
import prettytable
from tensorflow.python.keras.models import load_model
from utils.util_result import transform_predication, test_model, test_model_acc_fair, test_model_similar, \
    get_data_adult, get_average, draw_pie_chart, draw_Fairea_chart


def get_siamese_result_by_model(model_name, protected_name):
    """
    根据模型获取结果
    :return:
    """
    model_file = "adult_model/{}.h5".format(model_name)

    siamese_share = load_model(model_file)

    # 分离原始测试数据的保护属性，指定对应保护属性下的accurate fairness测试集
    test_file_org = "adult_data/test.txt"
    org_test_data, org_test_label = get_data_adult(test_file_org)
    org_test_pred = transform_predication(predicates=siamese_share.predict(org_test_data))
    if protected_name.find("age") >= 0:
        org_none_protect1, org_prot, org_none_protect2 = numpy.split(org_test_data, [10, 11], axis=1)
        test_file_aug = "adult_data/age_test.npz.npy"
    elif protected_name.find("race") >= 0:
        org_none_protect1, org_prot, org_none_protect2 = numpy.split(org_test_data, [11, 12], axis=1)
        test_file_aug = "adult_data/race_test.npz.npy"
    elif protected_name.find("sex") >= 0:
        org_none_protect1, org_prot, org_none_protect2 = numpy.split(org_test_data, [12, 13], axis=1)
        test_file_aug = "adult_data/sex_test.npz.npy"
    elif protected_name.find("multiple") >= 0:
        org_none_protect1, org_prot, org_none_protect2 = numpy.split(org_test_data, [10, 13], axis=1)
        test_file_aug = "adult_data/multiple_test.npz.npy"

    # 分离 Siamese fairness 测试数据的保护属性
    numpy_aug_test_data = numpy.load(test_file_aug)
    aug_test_data = []
    aug_test_label = []
    aug_test_pred = []
    aug_test_prot = []
    for i in range(numpy_aug_test_data.shape[1]):
        test_data, label = numpy.split(numpy_aug_test_data[:, i, :], [13, ], axis=1)
        predicate = transform_predication(predicates=siamese_share.predict(test_data))
        if protected_name.find("age") >= 0:
            none_protect1, aug_protect, none_protect2 = numpy.split(test_data, [10, 11], axis=1)
        elif protected_name.find("race") >= 0:
            none_protect1, aug_protect, none_protect2 = numpy.split(test_data, [11, 12], axis=1)
        elif protected_name.find("sex") >= 0:
            none_protect1, aug_protect, none_protect2 = numpy.split(test_data, [12, 13], axis=1)
        elif protected_name.find("multiple") >= 0:
            none_protect1, aug_protect, none_protect2 = numpy.split(test_data, [10, 13], axis=1)
        aug_test_data.append(test_data)
        aug_test_label.append(label)
        aug_test_pred.append(predicate)
        aug_test_prot.append(aug_protect)

    print("=================================================================================================")
    print(protected_name)
    # 测试 binary 保护属性的群体公平性
    if protected_name.find("sex") >= 0:
        # 原始测试集
        original_result = test_model(org_test_label, org_test_pred, org_prot)
        # Siamese fair 测试集
        similar_result = test_model_similar(org_test_label, aug_test_pred, aug_test_prot)
        other_metric_result = original_result + similar_result
        metric_table = prettytable.PrettyTable()
        metric_table.field_names = ["SPD", "EOD", "AVOD", "SPD_G", "EOD_G", "AVOD_G"]
        metric_table.add_row(other_metric_result)
        print(metric_table)
    # 测试当前保护属性下的 accurate fairness， 个体公平性
    AF_result = test_model_acc_fair(org_test_label, org_test_pred, aug_test_pred, org_test_data, 5)

    AF_table = prettytable.PrettyTable()
    AF_table.field_names = ["ACC", "Consistency", "IF", "TFR", "TBR", "FFR", "FBR", "F_R", "F_P", "F_F1"]
    AF_table.add_row(AF_result)
    print(AF_table)
    return AF_result


if __name__ == "__main__":
    print("======================================base line=================================================")
    "base line"
    protected_names = ["sex", "age", "race", "multiple"]
    bl_result = []
    for name in protected_names:
        bl_AF = get_siamese_result_by_model("BL_1_C", name)
        bl_result.append(bl_AF)
    print("==================================Siamese Fairness==============================================")
    "Siamese Fairness"
    protected_names = ["sex", "age", "race", "multiple"]
    sf_result = []
    for name in protected_names:
        sf_AF = get_siamese_result_by_model("FS_1_C_{}_share".format(name), name)
        sf_result.append(sf_AF)
    bl_acc = get_average(bl_result, 0)
    bl_fta = get_average(bl_result, 2)
    bl_tfr = get_average(bl_result, 3)
    bl_tbr = get_average(bl_result, 4)
    bl_ffr = get_average(bl_result, 5)
    bl_fbr = get_average(bl_result, 6)

    sf_acc = get_average(sf_result, 0)
    sf_fta = get_average(sf_result, 2)
    sf_tfr = get_average(sf_result, 3)
    sf_tbr = get_average(sf_result, 4)
    sf_ffr = get_average(sf_result, 5)
    sf_fbr = get_average(sf_result, 6)

    draw_pie_chart(TFR=bl_tfr, TBR=bl_tbr, FFR=bl_ffr, FBR=bl_fbr, name="adult_bl")
    draw_pie_chart(TFR=sf_tfr, TBR=sf_tbr, FFR=sf_ffr, FBR=sf_fbr, name="adult_sf")

    draw_Fairea_chart([bl_fta, sf_fta], [bl_acc, sf_acc], "adult_Fairea")

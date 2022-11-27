import numpy
import prettytable
from tensorflow.python.keras.models import load_model
from utils.util_result import transform_predication, test_model, test_model_acc_fair, test_model_similar, \
    get_data_compas, get_average, draw_pie_chart, draw_Fairea_chart


def get_siamese_result_by_model(model_name, protect_name):
    """
    根据模型获取结果
    :return:
    """

    model_file = "compas_model/{}.h5".format(model_name)
    model = load_model(model_file)

    # 分离原始测试数据的保护属性
    test_file_org = "compas_data/test.txt"
    org_test_data, org_test_label = get_data_compas(test_file_org)
    org_test_pred = transform_predication(predicates=model.predict(org_test_data))
    if protect_name.find("age") >= 0:
        org_none_protect1, org_prot, org_none_protect2 = numpy.split(org_test_data, [0, 1], axis=1)
        test_file_aug = "compas_data/age_test.npz.npy"
    elif protect_name.find("race") >= 0:
        org_none_protect1, org_prot, org_none_protect2 = numpy.split(org_test_data, [1, 2], axis=1)
        test_file_aug = "compas_data/race_test.npz.npy"
    elif protect_name.find("sex") >= 0:
        org_none_protect1, org_prot, org_none_protect2 = numpy.split(org_test_data, [2, 3], axis=1)
        test_file_aug = "compas_data/sex_test.npz.npy"
    elif protect_name.find("multiple") >= 0:
        org_none_protect1, org_prot, org_none_protect2 = numpy.split(org_test_data, [0, 3], axis=1)
        test_file_aug = "compas_data/multiple_test.npz.npy"

    # 分离 Siamese fairness 测试数据的保护属性
    numpy_aug_test_data = numpy.load(test_file_aug)
    aug_test_data = []
    aug_test_label = []
    aug_test_pred = []
    aug_test_prot = []
    for j in range(numpy_aug_test_data.shape[1]):
        test_data, label = numpy.split(numpy_aug_test_data[:, j, :], [11, ], axis=1)
        predicate = transform_predication(predicates=model.predict(test_data))
        if protect_name.find("age") >= 0:
            none_protect1, aug_protect, none_protect2 = numpy.split(test_data, [0, 1], axis=1)
        elif protect_name.find("race") >= 0:
            none_protect1, aug_protect, none_protect2 = numpy.split(test_data, [1, 2], axis=1)
        elif protect_name.find("sex") >= 0:
            none_protect1, aug_protect, none_protect2 = numpy.split(test_data, [2, 3], axis=1)
        elif protect_name.find("multiple") >= 0:
            none_protect1, aug_protect, none_protect2 = numpy.split(test_data, [0, 3], axis=1)
        aug_test_data.append(test_data)
        aug_test_label.append(label)
        aug_test_pred.append(predicate)
        aug_test_prot.append(aug_protect)

    print("=================================================================================================")
    print(protect_name)
    # 测试 binary 保护属性的群体公平性
    if protect_name.find("sex") >= 0:
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


# if __name__ == "__main__":
#     print("======================================base line=================================================")
#     "base line"
#     protected_names = ["sex", "age", "race", "multiple"]
#     for name in protected_names:
#         get_siamese_result_by_model("BL_1_C", name)
#     print("==================================Siamese Fairness==============================================")
#     "Siamese Fairness"
#     protected_names = ["sex", "age", "race", "multiple"]
#     for name in protected_names:
#         get_siamese_result_by_model("FS_1_C_{}_share".format(name), name)

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

    draw_pie_chart(TFR=bl_tfr, TBR=bl_tbr, FFR=bl_ffr, FBR=bl_fbr, name="compas_bl")
    draw_pie_chart(TFR=sf_tfr, TBR=sf_tbr, FFR=sf_ffr, FBR=sf_fbr, name="compas_sf")

    draw_Fairea_chart([bl_fta, sf_fta], [bl_acc, sf_acc], "adult_Fairea")

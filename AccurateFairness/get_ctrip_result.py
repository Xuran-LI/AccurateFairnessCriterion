import numpy
import prettytable
from tensorflow.python.keras.models import load_model
from utils.util_result import transform_predication, test_model_acc_fair, get_data_hotel, get_average, draw_pie_chart, \
    draw_Fairea_chart


def get_siamese_result_by_model(model_name):
    """
    根据模型获取结果
    :return:
    """

    model_file = "ctrip_model/{}.h5".format(model_name)
    siamese_share = load_model(model_file)

    # 分离原始测试数据的保护属性
    test_file_org = "ctrip_data/encode_test.txt"
    org_test_data, org_test_label = get_data_hotel(test_file_org)
    org_test_label = org_test_label[:, 2]
    org_test_pred = transform_predication(predicates=siamese_share.predict(org_test_data))
    org_none_protect1, org_prot = numpy.split(org_test_data, [6, ], axis=1)

    # 分离 Siamese fairness 测试数据的保护属性
    test_file_aug = "ctrip_data/multiple_test.npz.npy"
    numpy_aug_test_data = numpy.load(test_file_aug)
    aug_test_data = []
    aug_test_label = []
    aug_test_pred = []
    aug_test_prot = []
    for i in range(numpy_aug_test_data.shape[1]):
        test_data, label = numpy.split(numpy_aug_test_data[:, i, :], [12, ], axis=1)
        label = label[:, 2]
        predicate = transform_predication(predicates=siamese_share.predict(test_data))
        none_protect1, aug_protect = numpy.split(test_data, [6, ], axis=1)
        aug_test_data.append(test_data)
        aug_test_label.append(label)
        aug_test_pred.append(predicate)
        aug_test_prot.append(aug_protect)

    print("=================================================================================================")
    print("multiple customer consumption behaviour")

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
    bl_result = []
    bl_r = get_siamese_result_by_model("BL_1_C")
    bl_result.append(bl_r)
    print("==================================Siamese Fairness==============================================")
    "Siamese Fairness"
    sf_result = []
    sf_r = get_siamese_result_by_model("FS_1_C_multiple_share")
    sf_result.append(sf_r)

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

    draw_pie_chart(TFR=bl_tfr, TBR=bl_tbr, FFR=bl_ffr, FBR=bl_fbr, name="ctrip_bl")
    draw_pie_chart(TFR=sf_tfr, TBR=sf_tbr, FFR=sf_ffr, FBR=sf_fbr, name="ctrip_sf")

    draw_Fairea_chart([bl_fta, sf_fta], [bl_acc, sf_acc], "ctrip_Fairea")


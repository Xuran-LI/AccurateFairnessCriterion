import numpy
import tensorflow
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from tensorflow.python.keras.applications.densenet import layers
from tensorflow.python.keras.losses import MeanSquaredError, MeanAbsoluteError
from utils.util_result import get_data_hotel
from utils.utils_siamese_fair import SiameseFair_ModelCheckpoint, SiameseFair_Multiple


def get_siamese_classification_1(x, y, epochs, batch_size, protect_n, siamese_size):
    """
    使用多输入多输出模型训练，提升fairness accuracy
    :return:
    """
    # define the shared model
    input = keras.Input(shape=(12,), name="input")
    layer1 = layers.Dense(12, activation="relu")(input)
    layer2 = layers.Dense(4, activation="relu")(layer1)
    output = layers.Dense(4, activation="softmax")(layer2)
    model = keras.Model(inputs=input, outputs=output)
    model.summary()

    input_list = []
    output_list = []
    for j in range(siamese_size):
        input_j = keras.Input(shape=(12,), name="features_{}".format(j))
        output_j = model(input_j)
        input_list.append(input_j)
        output_list.append(output_j)
    # 保存最佳点
    checkpoint = SiameseFair_ModelCheckpoint("ctrip_model//FS_1_C_{}.h5".format(protect_n),
                                             monitor='performance_num', save_best_only=True)
    checkpoint.set_share_model(model, "ctrip_model/FS_1_C_{}_share.h5".format(protect_n))
    # 创建 FairSiamese_Multiple 模型
    siamese_model = SiameseFair_Multiple(inputs=input_list, outputs=output_list)
    siamese_model.set_parameter(siamese_size, MeanAbsoluteError(), MeanAbsoluteError())
    siamese_model.compile(loss=MeanSquaredError(), optimizer=tensorflow.keras.optimizers.Adam(), metrics=['acc'])
    history = siamese_model.fit(x=x, y=y, callbacks=[checkpoint], epochs=epochs, batch_size=batch_size, verbose=1)
    return history.history


def train_siamese_classification_1(siamese_file, protect_name):
    """
    训练siamese_classification
    :return:
    """
    train_x_list = []
    train_y_list = []

    train_file = "ctrip_data//encode_train.txt"
    x_train, y_train = get_data_hotel(filename=train_file)
    y_train = to_categorical(y_train[:, 2], num_classes=4)
    train_x_list.append(x_train)
    train_y_list.append(y_train)
    # 生成的 siamese_file
    aug_train = numpy.load(siamese_file)
    for i in range(aug_train.shape[1]):
        x_train, y_train = numpy.split(aug_train[:, i, :], [12, ], axis=1)
        y_train = to_categorical(y_train[:, 2], num_classes=4)
        train_x_list.append(x_train)
        train_y_list.append(y_train)

    get_siamese_classification_1(train_x_list, train_y_list, epochs, batch_size, protect_name, len(train_x_list))


epochs = 50
batch_size = 256
if __name__ == "__main__":
    aug_file = "ctrip_data/multiple_train.npz.npy"
    train_siamese_classification_1(aug_file, "multiple")

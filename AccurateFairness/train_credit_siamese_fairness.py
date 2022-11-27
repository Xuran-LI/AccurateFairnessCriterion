import numpy
import tensorflow
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from tensorflow.python.keras.applications.densenet import layers
from tensorflow.python.keras.losses import MeanSquaredError, MeanAbsoluteError

from utils.util_result import get_data_credit
from utils.utils_siamese_fair import SiameseFair_Multiple, SiameseFair_ModelCheckpoint


def get_siamese_classification(x, y, epochs, batch_size, protect_n, siamese_size):
    """
    使用多输入多输出模型训练，提升fairness accuracy
    :return:
    """
    # define the shared model
    model = keras.Sequential()
    model.add(layers.Dense(21, activation="relu"))
    model.add(layers.Dense(2, activation="relu"))
    model.add(layers.Dense(2, activation="softmax"))
    model.build(input_shape=(None, 21))
    model.summary()

    input_list = []
    output_list = []
    for j in range(siamese_size):
        input_j = keras.Input(shape=(21,), name="features_{}".format(j))
        output_j = model(input_j)
        input_list.append(input_j)
        output_list.append(output_j)
    # 保存最佳点
    checkpoint = SiameseFair_ModelCheckpoint("credit_model/FS_1_C_{}.h5".format(protect_n),
                                             monitor="performance_num", save_best_only=True)
    checkpoint.set_share_model(model, "credit_model/FS_1_C_{}_share.h5".format(protect_n))
    # 创建 FairSiamese_Multiple 模型
    siamese_model = SiameseFair_Multiple(inputs=input_list, outputs=output_list)
    siamese_model.set_parameter(siamese_size, MeanAbsoluteError(), MeanAbsoluteError())
    siamese_model.compile(loss=MeanSquaredError(), optimizer=tensorflow.keras.optimizers.Adam(), metrics=['acc'])
    history = siamese_model.fit(x=x, y=y, callbacks=[checkpoint], epochs=epochs, batch_size=batch_size, verbose=1)
    return history.history


def train_siamese_classification(siamese_file, protect_name):
    """
    训练siamese_classification
    :return:
    """
    epochs = 50
    batch_size = 32
    train_x_list = []
    train_y_list = []

    train_file = "credit_data/train.txt"
    x_train, y_train = get_data_credit(filename=train_file)
    y_train = to_categorical(y_train, num_classes=2)
    train_x_list.append(x_train)
    train_y_list.append(y_train)
    # 生成的 siamese_file
    aug_train = numpy.load(siamese_file)
    for i in range(aug_train.shape[1]):
        x_train, y_train = numpy.split(aug_train[:, i, :], [21, ], axis=1)
        y_train = to_categorical(y_train, num_classes=2)
        train_x_list.append(x_train)
        train_y_list.append(y_train)

    get_siamese_classification(train_x_list, train_y_list, epochs, batch_size, protect_name, len(train_x_list))


"训练Siamese Fairness Approach"
if __name__ == "__main__":
    "sex"
    sex_file = "credit_data/sex_train.npz.npy"
    train_siamese_classification(sex_file, "sex")
    "age"
    age_file = "credit_data/age_train.npz.npy"
    train_siamese_classification(age_file, "age")
    "multiple"
    multiple_file = "credit_data/multiple_train.npz.npy"
    train_siamese_classification(multiple_file, "multiple")

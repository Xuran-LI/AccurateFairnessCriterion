from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import tensorflow as tf
import six
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging


class SiameseFair_Multiple(tf.keras.Model):
    """
    生成keras.Model的子类，复现train_step方法，自定义loss损失函数
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.siamese_size = 2
        # initialize the lambda
        self.lambda_weight = None
        # initialize the D,d metric in Accurate Fairness
        self.distance_loss = None
        self.predication_loss = None

    def set_parameter(self, siamese_size, distance_loss, predication_loss):
        # set the number of shared model
        self.siamese_size = siamese_size
        # set the d,D metric in Accurate Fairness
        self.distance_loss = distance_loss
        self.predication_loss = predication_loss
        # initialize the lambda by siamese_size
        lambda_weight = OrderedDict()
        # 创建 lambda tensorflow.Variable 变量
        for i in range(siamese_size):
            lambda_weight["lambda_{}".format(i)] = tf.Variable(tf.random.normal(shape=()), name="lw_{}".format(i))
        self.lambda_weight = lambda_weight

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # 计算performance 损失
            # calculate the cumulative loss
            performance_list = []
            for i in range(self.siamese_size):
                performance_list.append(self.compiled_loss(y[0], y_pred[i]))
            # 计算预测距离
            # calculate the D
            pre_dis_list = []
            for i in range(self.siamese_size):
                pre_dis_list.append(self.predication_loss(y[0], y_pred[i]))
            # 计算属性距离
            # calculate the d
            fea_dis_list = []
            for i in range(self.siamese_size):
                fea_dis_list.append(self.distance_loss(x[0], x[i]))
            # calculate the cumulative loss
            performance_num = 0
            for j in range(self.siamese_size):
                performance_num += performance_list[j]
            # calculate the cumulative constraint by Accurate Fairness
            constraint_num = 0
            for j in range(self.siamese_size):
                constraint_num += (self.lambda_weight["lambda_{}".format(j)] * (pre_dis_list[j] - fea_dis_list[j]))
            # calculate the Accurate Fairness  loss
            total_loss = performance_num + constraint_num
        # 更新 lambda
        # update the lambda
        for l_w in range(self.siamese_size):
            new_lambda_weight = self.lambda_weight["lambda_{}".format(l_w)] + fea_dis_list[l_w] - pre_dis_list[l_w]
            zero = tf.zeros_like(new_lambda_weight)
            self.lambda_weight["lambda_{}".format(l_w)].assign(tf.where(new_lambda_weight > 0, new_lambda_weight, zero))

        # 更新模型参数loss
        # update the shared model parameter
        self.optimizer.minimize(total_loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y[0], y_pred[0], sample_weight)
        return_metrics = {"performance_num": performance_num}
        return return_metrics


class SiameseFair_ModelCheckpoint(ModelCheckpoint):
    def __int__(self):
        super(SiameseFair_ModelCheckpoint, self).__int__()
        self.share_model = None
        self.share_model_path = None

    def set_share_model(self, share_model, share_model_path):
        self.share_model = share_model
        self.share_model_path = share_model_path

    def _save_model(self, epoch, logs):
        logs = logs or {}

        if isinstance(self.save_freq, int) or self.epochs_since_last_save >= self.period:
            logs = tf_utils.to_numpy_or_python_type(logs)
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, logs)

            try:
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        logging.warning(
                            'Can save best model only with %s available,  skipping.', self.monitor)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,  saving model to %s' % (
                                    epoch + 1, self.monitor, self.best, current, filepath))
                            self.best = current
                            if self.save_weights_only:
                                self.model.save_weights(filepath, overwrite=True, options=self._options)
                                self.share_model.save_weights(self.share_model_path, overwrite=True,
                                                              options=self._options)
                            else:
                                self.model.save(filepath, overwrite=True, options=self._options)
                                self.share_model.save(self.share_model_path, overwrite=True, options=self._options)
                        else:
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s did not improve from %0.5f' % (
                                    epoch + 1, self.monitor, self.best))
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True, options=self._options)
                        self.share_model.save_weights(self.share_model_path, overwrite=True, options=self._options)
                    else:
                        self.model.save(filepath, overwrite=True, options=self._options)
                        self.share_model.save(self.share_model_path, overwrite=True, options=self._options)
                self._maybe_remove_file()
            except IOError as e:
                if 'is a directory' in six.ensure_str(e.args[0]).lower():
                    raise IOError('Please specify a non-directory filepath for ModelCheckpoint.'
                                  ' Filepath used is an existing directory: {}'.format(filepath))
                raise e

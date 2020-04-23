import numpy as np
import matplotlib.pyplot as plt  # для построения графиков
from PIL import Image  # для работы с изображениями
import os  # для работы с файлами на диске
import model  # книга с функциями
import random
import glob
import cv2 as cv
import time
import make_data as md


class CNN:
    def __init__(self, train=False):
        self.train_model = train

        # параметры для первого слоя конволюции (начальные параметры будут инициализированы во время работы сети)
        # веса для дообучения сети будут подгружены из файла
        self.conv_w_1 = []
        self.conv_b_1 = []
        # параметры для второго слоя конволюции
        self.conv_w_2 = []
        self.conv_b_2 = []
        # параметры для первого слоя fc-сети
        self.fc_w_1 = np.array([[]])
        self.fc_b_1 = np.array([[]])
        # параметры для второго слоя fc-сети
        self.fc_w_2 = np.array([[]])
        self.fc_b_2 = np.array([[]])

        self.loss_change = []
        self.accuracy_change = []

        self.weight_dir = '/cnn_weights.npy'  # веса обученной сети
        self.trainX = []
        self.testX = []
        self.trainY = []
        self.testY = []
        self.model_settings = {'learning_rate': 0.01,  # коэффициент обучения
                               'conv_shape_1': (2, 2),  # размер ядра свертки
                               'conv_shape_2': (3, 3),
                               'maxpool_shape_1': (2, 2),  # размер окна макспулинга
                               'conv_feature_1': 5,  # количесвто feature maps на выходе функции
                               'conv_feature_2': 20,
                               'conv_stride_1': 2,  # величина шага
                               'conv_stride_2': 1,
                               'maxpool_stride_1': 2,
                               'fc_neurons_1': 2000,  # количество нейронов в скрытом слое
                               'conv_fn_1': 'relu',  # функция активации
                               'conv_fn_2': 'sigmoid',
                               'fc_fn_1': 'sigmoid',
                               'fc_fn_2': 'softmax',
                               'conv_conv_1': False,  # операция конволюции или кросс-корреляции
                               'conv_conv_2': False,
                               'maxpool_conv_1': False,  # "конволюция" или "корреляция" для операции макспулинга
                               'conv_center_1': (0, 0),  # центр ядра
                               'conv_center_2': (1, 1),
                               'maxpool_center_1': (0, 0)}

    def load_data(self):
        (self.trainX, self.testX, self.trainY, self.testY) = md.read_data_sets()

    '''
    def load_weights(self):
        if not os.path.isfile(self.weight_dir):
            w1 = np.random.normal(scale=0.1, size=(2, 2, 1, 5))
            w1 = np.reshape(w1, (w1.size,))
            w1 = np.reshape(w1, (5, 2, 2), order='F')
            for i in range(5):
                self.conv_w_1.append(w1[i].T)
            w3 = np.random.normal(scale=0.1, size=(3, 3, 5, 20))
            w3 = np.reshape(w3, (w3.size,))
            w3 = np.reshape(w3, (5 * 20, 3, 3), order='F')
            for i in range(5 * 20):
                self.conv_w_2.append(w3[i].T)
            self.conv_b_1 = [0.1 for _ in range(5)]
            self.conv_b_2 = [0.1 for _ in range(20)]
            self.fc_w_1 = np.random.normal(scale=0.1, size=(7 * 7 * 20, 2000))
            self.fc_b_1 = [0.1 for _ in range(2000)]
            self.fc_w_2 = np.random.normal(scale=0.1, size=(2000, 10))
            self.fc_b_2 = [0.1 for _ in range(10)]
    '''

    def training(self):
        # первый и последний шаги
        if self.train_model:
            start_step = model.get_start_step(self.weight_dir)
            end_step = len(self.trainX)
            len_dataset = 10  # частота вывода print и сохранения весов (не менять при возобновлении обучения)
        else:
            start_step = 0
            end_step = len(self.trainX)
            len_dataset = 10

        # self.load_weights()

        if self.train_model:
            self.loss_change = model.get_saved('loss_change', self.weight_dir)
            self.accuracy_change = model.get_saved('accuracy_change', self.weight_dir)
        else:
            self.loss_change = []
            self.accuracy_change = []

        for step in range(start_step, end_step):
            # извлечение изображения из хранилища
            image_id = step % len(self.trainX)  # на каждом шаге обновляются веса для одного изображения
            print('до вывода результатов', str(round((step % len_dataset) * 100 / len_dataset)) + '%', end="\r")
            input_image = [self.trainX[image_id]]  # здесь лист, так как convolution_feed на
            # вход принимает лист, состоящий из feature maps
            y_true = self.testY[image_id]
            # прямое прохожение сети
            # первый конволюционный слой
            conv_y_1, conv_w_1, conv_b_1 = model.convolution_feed(
                y_l_minus_1=input_image,
                w_l=self.conv_w_1,
                w_l_name='conv_w_1',  # для подгрузки весов из файла
                w_shape_l=self.model_settings['conv_shape_1'],
                b_l=self.conv_b_1,
                b_l_name='conv_b_1',
                feature_maps=self.model_settings['conv_feature_1'],
                act_fn=self.model_settings['conv_fn_1'],
                dir_npy=self.weight_dir,
                conv_params={
                    'convolution': self.model_settings['conv_conv_1'],
                    'stride': self.model_settings['conv_stride_1'],
                    'center_w_l': self.model_settings['conv_center_1']
                }
            )
            # слой макспулинга
            conv_y_1_mp, conv_y_1_mp_to_conv_y_1 = model.maxpool_feed(
                y_l=conv_y_1,
                conv_params={
                    'window_shape': self.model_settings['maxpool_shape_1'],
                    'convolution': self.model_settings['maxpool_conv_1'],
                    'stride': self.model_settings['maxpool_stride_1'],
                    'center_window': self.model_settings['maxpool_center_1']
                }
            )
            # второй конволюционный слой
            conv_y_2, conv_w_2, conv_b_2 = model.convolution_feed(
                y_l_minus_1=conv_y_1_mp,
                w_l=self.conv_w_2,
                w_l_name='conv_w_2',
                w_shape_l=self.model_settings['conv_shape_2'],
                b_l=self.conv_b_2,
                b_l_name='conv_b_2',
                feature_maps=self.model_settings['conv_feature_2'],
                act_fn=self.model_settings['conv_fn_2'],
                dir_npy=self.weight_dir,
                conv_params={
                    'convolution': self.model_settings['conv_conv_2'],
                    'stride': self.model_settings['conv_stride_2'],
                    'center_w_l': self.model_settings['conv_center_2']
                }
            )
            # конвертация полученных feature maps в вектор
            conv_y_2_vect = model.matrix2vector_tf(conv_y_2)
            # первый слой fully connected сети
            fc_y_1, fc_w_1, fc_b_1 = model.fc_multiplication(
                y_l_minus_1=conv_y_2_vect,
                w_l=self.fc_w_1,
                w_l_name='fc_w_1',
                b_l=self.fc_b_1,
                b_l_name='fc_b_1',
                neurons=self.model_settings['fc_neurons_1'],
                act_fn=self.model_settings['fc_fn_1'],
                dir_npy=self.weight_dir
            )
            # второй слой fully connected сети
            fc_y_2, fc_w_2, fc_b_2 = model.fc_multiplication(
                y_l_minus_1=fc_y_1,
                w_l=self.fc_w_2,
                w_l_name='fc_w_2',
                b_l=self.fc_b_2,
                b_l_name='fc_b_2',
                neurons=self.trainY.max(),  # количество нейронов на выходе моледи равно числу классов
                act_fn=self.model_settings['fc_fn_2'],
                dir_npy=self.weight_dir
            )
            # ошибка модели
            fc_error = model.loss_fn(y_true, fc_y_2, feed=True)
            # сохранение значений loss и accuracy
            self.loss_change.append(fc_error.sum())
            self.accuracy_change.append(y_true.argmax() == fc_y_2.argmax())
            # обратное прохожение по сети
            if self.train_model:
                # backprop через loss-функцию
                dEdfc_y_2 = model.loss_fn(y_true, fc_y_2, feed=False)
                # backprop через второй слой fc-сети
                dEdfc_y_1, fc_w_2, fc_b_2 = model.fc_backpropagation(
                    y_l_minus_1=fc_y_1,
                    dEdy_l=dEdfc_y_2,
                    y_l=fc_y_2,
                    w_l=fc_w_2,
                    b_l=fc_b_2,
                    act_fn=self.model_settings['fc_fn_2'],
                    alpha=self.model_settings['learning_rate']
                )
                # backprop через первый слой fc-сети
                dEdfc_y_0, fc_w_1, fc_b_1 = model.fc_backpropagation(
                    y_l_minus_1=conv_y_2_vect,
                    dEdy_l=dEdfc_y_1,
                    y_l=fc_y_1,
                    w_l=fc_w_1,
                    b_l=fc_b_1,
                    act_fn=self.model_settings['fc_fn_1'],
                    alpha=self.model_settings['learning_rate']
                )
                # конвертация полученного вектора в feature maps
                dEdconv_y_2 = model.vector2matrix_tf(
                    vector=dEdfc_y_0,
                    matrix_shape=conv_y_2[0].shape  # размерность одной из матриц feature map
                )
                # backprop через второй слой конволюции
                dEdconv_y_1_mp, conv_w_2, conv_b_2 = model.convolution_backpropagation(
                    y_l_minus_1=conv_y_1_mp,  # так как слой макспулинга!
                    y_l=conv_y_2,
                    w_l=conv_w_2,
                    b_l=conv_b_2,
                    dEdy_l=dEdconv_y_2,
                    feature_maps=self.model_settings['conv_feature_2'],
                    act_fn=self.model_settings['conv_fn_2'],
                    alpha=self.model_settings['learning_rate'],
                    conv_params={
                        'convolution': self.model_settings['conv_conv_2'],
                        'stride': self.model_settings['conv_stride_2'],
                        'center_w_l': self.model_settings['conv_center_2']
                    }
                )
                # backprop через слой макспулинга
                dEdconv_y_1 = model.maxpool_back(
                    dEdy_l_mp=dEdconv_y_1_mp,
                    y_l_mp_to_y_l=conv_y_1_mp_to_conv_y_1,
                    y_l_shape=conv_y_1[0].shape
                )
                # backprop через первый слой конволюции
                dEdconv_y_0, conv_w_1, conv_b_1 = model.convolution_backpropagation(
                    y_l_minus_1=input_image,
                    y_l=conv_y_1,
                    w_l=conv_w_1,
                    b_l=conv_b_1,
                    dEdy_l=dEdconv_y_1,
                    feature_maps=self.model_settings['conv_feature_1'],
                    act_fn=self.model_settings['conv_fn_1'],
                    alpha=self.model_settings['learning_rate'],
                    conv_params={
                        'convolution': self.model_settings['conv_conv_1'],
                        'stride': self.model_settings['conv_stride_1'],
                        'center_w_l': self.model_settings['conv_center_1']
                    }
                )
            # вывод результатов
            if len(self.loss_change) % len_dataset == 0:
                print('шаг:', len(self.loss_change), 'loss:', sum(self.loss_change[-len_dataset:]) / len_dataset, 'accuracy:',
                      sum(self.accuracy_change[-len_dataset:]) / len_dataset)
                # сохранение весов
                if self.train_model:
                    np.save(self.weight_dir, {
                        'step': step,
                        'loss_change': self.loss_change,
                        'accuracy_change': self.accuracy_change,
                        'conv_w_1': conv_w_1,
                        'conv_b_1': conv_b_1,
                        'conv_w_2': conv_w_2,
                        'conv_b_2': conv_b_2,
                        'fc_w_1': fc_w_1,
                        'fc_b_1': fc_b_1,
                        'fc_w_2': fc_w_2,
                        'fc_b_2': fc_b_2
                    }
                            )

        if not self.train_model:
            print('test_loss:', sum(self.loss_change) / len(self.loss_change), 'test_accuracy:',
                  sum(self.accuracy_change) / len(self.accuracy_change))

def main():
     network = CNN()
     network.load_data()
     network.training()


if __name__ == "__main__":
    main()

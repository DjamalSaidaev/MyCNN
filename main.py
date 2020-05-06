# -*- coding: utf-8 -*-

import numpy as np
import os  # для работы с файлами на диске
import model  # книга с функциями
import time


class CNN:
    def __init__(self):
        self.classes = {
            "1": "B",
            "2": "P",
            "3": "R",
            "4": "S",
            "5": "Ни на что не похоже",
        }
        # параметры для первого слоя конволюции (начальные параметры будут инициализированы во время работы сети)
        # веса для дообучения сети будут подгружены из файла
        self.conv_w_1 = np.array([])
        self.conv_b_1 = np.array([])
        # параметры для второго слоя конволюции
        self.conv_w_2 = np.array([])
        self.conv_b_2 = np.array([])
        # параметры для первого слоя fc-сети
        self.fc_w_1 = np.array([[]])
        self.fc_b_1 = np.array([[]])
        # параметры для второго слоя fc-сети
        self.fc_w_2 = np.array([[]])
        self.fc_b_2 = np.array([[]])

        self.loss_change = []
        self.accuracy_change = []

        self.test_loss_change = []
        self.test_accuracy_change = []

        self.weight_dir = None
        self.trainX = []
        self.testX = []
        self.trainY = []
        self.testY = []
        self.epochs = None
        self.model_settings = {
            "learning_rate": 0.01,  # коэффициент обучения
            "conv_shape_1": (2, 2),  # размер ядра свертки
            "conv_shape_2": (3, 3),
            "maxpool_shape_1": (2, 2),  # размер окна макспулинга
            "conv_feature_1": 5,  # количесвто feature maps на выходе функции
            "conv_feature_2": 20,
            "conv_stride_1": 2,  # величина шага
            "conv_stride_2": 1,
            "maxpool_stride_1": 2,
            "fc_neurons_1": 2000,  # количество нейронов в скрытом слое
            "conv_fn_1": "relu",  # функция активации
            "conv_fn_2": "sigmoid",
            "fc_fn_1": "sigmoid",
            "fc_fn_2": "softmax",
            "conv_conv_1": False,  # операция конволюции или кросс-корреляции
            "conv_conv_2": False,
            "maxpool_conv_1": False,  # "конволюция" или "корреляция" для операции макспулинга
            "conv_center_1": (0, 0),  # центр ядра
            "conv_center_2": (1, 1),
            "maxpool_center_1": (0, 0),
        }

    def load_data_from_file(self, file):
        (self.trainX, self.testX, self.trainY, self.testY) = (
            np.load(file, allow_pickle=True).item().get("data")
        )

    def load_model(self, path):
        self.weight_dir = path
        self.conv_w_1 = (
            np.load(self.weight_dir, allow_pickle=True).item().get("conv_w_1")
        )
        self.conv_b_1 = (
            np.load(self.weight_dir, allow_pickle=True).item().get("conv_b_1")
        )

        self.conv_w_2 = (
            np.load(self.weight_dir, allow_pickle=True).item().get("conv_w_2")
        )
        self.conv_b_2 = (
            np.load(self.weight_dir, allow_pickle=True).item().get("conv_b_2")
        )

        self.fc_w_1 = np.load(self.weight_dir, allow_pickle=True).item().get("fc_w_1")
        self.fc_b_1 = np.load(self.weight_dir, allow_pickle=True).item().get("fc_b_1")

        self.fc_w_2 = np.load(self.weight_dir, allow_pickle=True).item().get("fc_w_2")
        self.fc_b_2 = np.load(self.weight_dir, allow_pickle=True).item().get("fc_b_2")
        print("Модель загружена")

    def gaussian_blur(self, img):
        kernel = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])
        kernel = kernel / np.sum(kernel)
        arraylist = []
        for y in range(3):
            temparray = np.copy(img)
            temparray = np.roll(temparray, y - 1, axis=0)
            for x in range(3):
                temparray_X = np.copy(temparray)
                temparray_X = np.roll(temparray_X, x - 1, axis=1) * kernel[y, x]
                arraylist.append(temparray_X)

        arraylist = np.array(arraylist)
        arraylist_sum = np.sum(arraylist, axis=0)
        return arraylist_sum

    def input_image_by_matrix(self):
        print("Введите изображение (8x8) построчно:")
        matrix = []
        for _ in range(8):
            temp = input().split()
            temp = [int(t) for t in temp]
            matrix.append(temp)
        try:
            matrix = np.array(matrix, dtype="float")
            matrix = self.gaussian_blur(matrix)
        except:
            print("Неверный ввод")
            return matrix
        return matrix

    def predict(self, path):
        img = path
        input_image = [img]
        conv_y_1, self.conv_w_1, self.conv_b_1 = model.convolution_feed(
            y_l_minus_1=input_image,
            w_l=self.conv_w_1,
            w_l_name="conv_w_1",  # для подгрузки весов из файла
            w_shape_l=self.model_settings["conv_shape_1"],
            b_l=self.conv_b_1,
            b_l_name="conv_b_1",
            feature_maps=self.model_settings["conv_feature_1"],
            act_fn=self.model_settings["conv_fn_1"],
            dir_npy=self.weight_dir,
            conv_params={
                "convolution": self.model_settings["conv_conv_1"],
                "stride": self.model_settings["conv_stride_1"],
                "center_w_l": self.model_settings["conv_center_1"],
            },
        )
        # слой макспулинга
        conv_y_1_mp, conv_y_1_mp_to_conv_y_1 = model.maxpool_feed(
            y_l=conv_y_1,
            conv_params={
                "window_shape": self.model_settings["maxpool_shape_1"],
                "convolution": self.model_settings["maxpool_conv_1"],
                "stride": self.model_settings["maxpool_stride_1"],
                "center_window": self.model_settings["maxpool_center_1"],
            },
        )
        # второй конволюционный слой
        conv_y_2, self.conv_w_2, self.conv_b_2 = model.convolution_feed(
            y_l_minus_1=conv_y_1_mp,
            w_l=self.conv_w_2,
            w_l_name="conv_w_2",
            w_shape_l=self.model_settings["conv_shape_2"],
            b_l=self.conv_b_2,
            b_l_name="conv_b_2",
            feature_maps=self.model_settings["conv_feature_2"],
            act_fn=self.model_settings["conv_fn_2"],
            dir_npy=self.weight_dir,
            conv_params={
                "convolution": self.model_settings["conv_conv_2"],
                "stride": self.model_settings["conv_stride_2"],
                "center_w_l": self.model_settings["conv_center_2"],
            },
        )
        # конвертация полученных feature maps в вектор
        conv_y_2_vect = model.matrix2vector_tf(conv_y_2)
        # первый слой fully connected сети
        fc_y_1, self.fc_w_1, self.fc_b_1 = model.fc_multiplication(
            y_l_minus_1=conv_y_2_vect,
            w_l=self.fc_w_1,
            w_l_name="fc_w_1",
            b_l=self.fc_b_1,
            b_l_name="fc_b_1",
            neurons=self.model_settings["fc_neurons_1"],
            act_fn=self.model_settings["fc_fn_1"],
            dir_npy=self.weight_dir,
        )
        # второй слой fully connected сети
        fc_y_2, self.fc_w_2, self.fc_b_2 = model.fc_multiplication(
            y_l_minus_1=fc_y_1,
            w_l=self.fc_w_2,
            w_l_name="fc_w_2",
            b_l=self.fc_b_2,
            b_l_name="fc_b_2",
            neurons=5,  # количество нейронов на выходе моледи равно числу классов
            act_fn=self.model_settings["fc_fn_2"],
            dir_npy=self.weight_dir,
        )
        print("Вероятность совпадения с образцами:")
        return [(self.classes[i], fc_y_2[0][int(i) - 1]) for i in self.classes.keys()]

    def save_model(self, path):
        np.save(
            path,
            {
                "epochs": self.epochs,
                "len_train": len(self.trainX),
                "test_loss": self.test_loss_change,
                "test_accuracy": self.test_accuracy_change,
                "loss_change": self.loss_change,
                "accuracy_change": self.accuracy_change,
                "conv_w_1": self.conv_w_1,
                "conv_b_1": self.conv_b_1,
                "conv_w_2": self.conv_w_2,
                "conv_b_2": self.conv_b_2,
                "fc_w_1": self.fc_w_1,
                "fc_b_1": self.fc_b_1,
                "fc_w_2": self.fc_w_2,
                "fc_b_2": self.fc_b_2,
            },
        )

    def testing(self):
        cur_time = time.time()
        start_step = 0
        end_step = len(self.testX)
        self.test_loss_change = []
        self.test_accuracy_change = []

        for step in range(start_step, end_step):
            # извлечение изображения из хранилища
            input_image = [self.testX[step]]  # здесь лист, так как convolution_feed на
            # вход принимает лист, состоящий из feature maps
            y_true = np.array([0 for _ in range(5)])
            y_true[self.testY[step] - 1] = 1
            # прямое прохожение сети
            # первый конволюционный слой
            conv_y_1, self.conv_w_1, self.conv_b_1 = model.convolution_feed(
                y_l_minus_1=input_image,
                w_l=self.conv_w_1,
                w_l_name="conv_w_1",  # для подгрузки весов из файла
                w_shape_l=self.model_settings["conv_shape_1"],
                b_l=self.conv_b_1,
                b_l_name="conv_b_1",
                feature_maps=self.model_settings["conv_feature_1"],
                act_fn=self.model_settings["conv_fn_1"],
                dir_npy=self.weight_dir,
                conv_params={
                    "convolution": self.model_settings["conv_conv_1"],
                    "stride": self.model_settings["conv_stride_1"],
                    "center_w_l": self.model_settings["conv_center_1"],
                },
            )
            # слой макспулинга
            conv_y_1_mp, conv_y_1_mp_to_conv_y_1 = model.maxpool_feed(
                y_l=conv_y_1,
                conv_params={
                    "window_shape": self.model_settings["maxpool_shape_1"],
                    "convolution": self.model_settings["maxpool_conv_1"],
                    "stride": self.model_settings["maxpool_stride_1"],
                    "center_window": self.model_settings["maxpool_center_1"],
                },
            )
            # второй конволюционный слой
            conv_y_2, self.conv_w_2, self.conv_b_2 = model.convolution_feed(
                y_l_minus_1=conv_y_1_mp,
                w_l=self.conv_w_2,
                w_l_name="conv_w_2",
                w_shape_l=self.model_settings["conv_shape_2"],
                b_l=self.conv_b_2,
                b_l_name="conv_b_2",
                feature_maps=self.model_settings["conv_feature_2"],
                act_fn=self.model_settings["conv_fn_2"],
                dir_npy=self.weight_dir,
                conv_params={
                    "convolution": self.model_settings["conv_conv_2"],
                    "stride": self.model_settings["conv_stride_2"],
                    "center_w_l": self.model_settings["conv_center_2"],
                },
            )
            # конвертация полученных feature maps в вектор
            conv_y_2_vect = model.matrix2vector_tf(conv_y_2)
            # первый слой fully connected сети
            fc_y_1, self.fc_w_1, self.fc_b_1 = model.fc_multiplication(
                y_l_minus_1=conv_y_2_vect,
                w_l=self.fc_w_1,
                w_l_name="fc_w_1",
                b_l=self.fc_b_1,
                b_l_name="fc_b_1",
                neurons=self.model_settings["fc_neurons_1"],
                act_fn=self.model_settings["fc_fn_1"],
                dir_npy=self.weight_dir,
            )
            # второй слой fully connected сети
            fc_y_2, self.fc_w_2, self.fc_b_2 = model.fc_multiplication(
                y_l_minus_1=fc_y_1,
                w_l=self.fc_w_2,
                w_l_name="fc_w_2",
                b_l=self.fc_b_2,
                b_l_name="fc_b_2",
                neurons=5,  # количество нейронов на выходе моледи равно числу классов
                act_fn=self.model_settings["fc_fn_2"],
                dir_npy=self.weight_dir,
            )
            # ошибка модели
            fc_error = model.loss_fn(y_true, fc_y_2, feed=True)
            # сохранение значений loss и accuracy
            self.test_loss_change.append(fc_error.sum())
            self.test_accuracy_change.append(y_true.argmax() == fc_y_2.argmax())
        print(
            "Тестирование модели\n loss: {} accuracy: {}".format(
                sum(self.test_loss_change) / len(self.test_loss_change),
                sum(self.test_accuracy_change) / len(self.test_accuracy_change),
            )
        )
        cur_time = time.time() - cur_time
        return cur_time

    def training(self, epochs=30, save="train.npy"):
        cur_time = time.time()
        self.weight_dir = os.path.join(os.path.dirname(__file__), save)
        self.epochs = epochs
        start_step = 0
        end_step = len(self.trainX)

        self.loss_change = []
        self.accuracy_change = []
        for epoch in range(1, epochs + 1):
            for step in range(start_step, end_step):
                # извлечение изображения из хранилища
                input_image = [
                    self.trainX[step]
                ]  # здесь лист, так как convolution_feed на
                # вход принимает лист, состоящий из feature maps
                y_true = np.array([0 for _ in range(5)])
                y_true[self.trainY[step] - 1] = 1
                # прямое прохожение сети
                # первый конволюционный слой
                conv_y_1, self.conv_w_1, self.conv_b_1 = model.convolution_feed(
                    y_l_minus_1=input_image,
                    w_l=self.conv_w_1,
                    w_l_name="conv_w_1",  # для подгрузки весов из файла
                    w_shape_l=self.model_settings["conv_shape_1"],
                    b_l=self.conv_b_1,
                    b_l_name="conv_b_1",
                    feature_maps=self.model_settings["conv_feature_1"],
                    act_fn=self.model_settings["conv_fn_1"],
                    dir_npy=self.weight_dir,
                    conv_params={
                        "convolution": self.model_settings["conv_conv_1"],
                        "stride": self.model_settings["conv_stride_1"],
                        "center_w_l": self.model_settings["conv_center_1"],
                    },
                )
                # слой макспулинга
                conv_y_1_mp, conv_y_1_mp_to_conv_y_1 = model.maxpool_feed(
                    y_l=conv_y_1,
                    conv_params={
                        "window_shape": self.model_settings["maxpool_shape_1"],
                        "convolution": self.model_settings["maxpool_conv_1"],
                        "stride": self.model_settings["maxpool_stride_1"],
                        "center_window": self.model_settings["maxpool_center_1"],
                    },
                )
                # второй конволюционный слой
                conv_y_2, self.conv_w_2, self.conv_b_2 = model.convolution_feed(
                    y_l_minus_1=conv_y_1_mp,
                    w_l=self.conv_w_2,
                    w_l_name="conv_w_2",
                    w_shape_l=self.model_settings["conv_shape_2"],
                    b_l=self.conv_b_2,
                    b_l_name="conv_b_2",
                    feature_maps=self.model_settings["conv_feature_2"],
                    act_fn=self.model_settings["conv_fn_2"],
                    dir_npy=self.weight_dir,
                    conv_params={
                        "convolution": self.model_settings["conv_conv_2"],
                        "stride": self.model_settings["conv_stride_2"],
                        "center_w_l": self.model_settings["conv_center_2"],
                    },
                )
                # конвертация полученных feature maps в вектор
                conv_y_2_vect = model.matrix2vector_tf(conv_y_2)
                # первый слой fully connected сети
                fc_y_1, self.fc_w_1, self.fc_b_1 = model.fc_multiplication(
                    y_l_minus_1=conv_y_2_vect,
                    w_l=self.fc_w_1,
                    w_l_name="fc_w_1",
                    b_l=self.fc_b_1,
                    b_l_name="fc_b_1",
                    neurons=self.model_settings["fc_neurons_1"],
                    act_fn=self.model_settings["fc_fn_1"],
                    dir_npy=self.weight_dir,
                )
                # второй слой fully connected сети
                fc_y_2, self.fc_w_2, self.fc_b_2 = model.fc_multiplication(
                    y_l_minus_1=fc_y_1,
                    w_l=self.fc_w_2,
                    w_l_name="fc_w_2",
                    b_l=self.fc_b_2,
                    b_l_name="fc_b_2",
                    neurons=5,  # количество нейронов на выходе моледи равно числу классов
                    act_fn=self.model_settings["fc_fn_2"],
                    dir_npy=self.weight_dir,
                )
                # ошибка модели
                fc_error = model.loss_fn(y_true, fc_y_2, feed=True)
                # сохранение значений loss и accuracy
                self.loss_change.append(fc_error.sum())
                self.accuracy_change.append(y_true.argmax() == fc_y_2.argmax())

                # обратное прохожение по сети
                # backprop через loss-функцию
                dEdfc_y_2 = model.loss_fn(y_true, fc_y_2, feed=False)
                # backprop через второй слой fc-сети
                dEdfc_y_1, self.fc_w_2, self.fc_b_2 = model.fc_backpropagation(
                    y_l_minus_1=fc_y_1,
                    dEdy_l=dEdfc_y_2,
                    y_l=fc_y_2,
                    w_l=self.fc_w_2,
                    b_l=self.fc_b_2,
                    act_fn=self.model_settings["fc_fn_2"],
                    alpha=self.model_settings["learning_rate"],
                )
                # backprop через первый слой fc-сети
                dEdfc_y_0, self.fc_w_1, self.fc_b_1 = model.fc_backpropagation(
                    y_l_minus_1=conv_y_2_vect,
                    dEdy_l=dEdfc_y_1,
                    y_l=fc_y_1,
                    w_l=self.fc_w_1,
                    b_l=self.fc_b_1,
                    act_fn=self.model_settings["fc_fn_1"],
                    alpha=self.model_settings["learning_rate"],
                )
                # конвертация полученного вектора в feature maps
                dEdconv_y_2 = model.vector2matrix_tf(
                    vector=dEdfc_y_0,
                    matrix_shape=conv_y_2[
                        0
                    ].shape,  # размерность одной из матриц feature map
                )
                # backprop через второй слой конволюции
                (
                    dEdconv_y_1_mp,
                    self.conv_w_2,
                    self.conv_b_2,
                ) = model.convolution_backpropagation(
                    y_l_minus_1=conv_y_1_mp,  # так как слой макспулинга!
                    y_l=conv_y_2,
                    w_l=self.conv_w_2,
                    b_l=self.conv_b_2,
                    dEdy_l=dEdconv_y_2,
                    feature_maps=self.model_settings["conv_feature_2"],
                    act_fn=self.model_settings["conv_fn_2"],
                    alpha=self.model_settings["learning_rate"],
                    conv_params={
                        "convolution": self.model_settings["conv_conv_2"],
                        "stride": self.model_settings["conv_stride_2"],
                        "center_w_l": self.model_settings["conv_center_2"],
                    },
                )
                # backprop через слой макспулинга
                dEdconv_y_1 = model.maxpool_back(
                    dEdy_l_mp=dEdconv_y_1_mp,
                    y_l_mp_to_y_l=conv_y_1_mp_to_conv_y_1,
                    y_l_shape=conv_y_1[0].shape,
                )
                # backprop через первый слой конволюции
                (
                    dEdconv_y_0,
                    self.conv_w_1,
                    self.conv_b_1,
                ) = model.convolution_backpropagation(
                    y_l_minus_1=input_image,
                    y_l=conv_y_1,
                    w_l=self.conv_w_1,
                    b_l=self.conv_b_1,
                    dEdy_l=dEdconv_y_1,
                    feature_maps=self.model_settings["conv_feature_1"],
                    act_fn=self.model_settings["conv_fn_1"],
                    alpha=self.model_settings["learning_rate"],
                    conv_params={
                        "convolution": self.model_settings["conv_conv_1"],
                        "stride": self.model_settings["conv_stride_1"],
                        "center_w_l": self.model_settings["conv_center_1"],
                    },
                )
                # вывод результатов
                print(
                    "эпоха: {} шаг: {} loss: {:.3f} accuracy: {:.3f}".format(
                        epoch,
                        step,
                        sum(self.loss_change) / len(self.loss_change),
                        sum(self.accuracy_change) / len(self.accuracy_change),
                    )
                )
                # перемешивание тренировочных данных
                model.shuffle_list(self.trainX, self.trainY)
        # тестирование и сохранение модели
        cur_time = time.time() - cur_time
        self.save_model(self.weight_dir)
        return cur_time, self.testing()


def main():
    network = CNN()
    train = False
    weight_dir = os.path.join(os.path.dirname(__file__), "cnn_weights_epam.npy")
    data_dir = os.path.join(os.path.dirname(__file__), "train_test_data.npy")
    if train:
        network.load_data_from_file(data_dir)
        train_time, test_time = network.training(epochs=10, save=weight_dir)
        print(
            "Время на обучение: {:.2f} мин. \n"
            "Время на тестирование модели: {:.2f} мин".format(
                train_time / 60.0, test_time / 60.0
            )
        )
        model.draw_history_of_training(weight_dir)
    else:
        network.load_model(weight_dir)
        for obj in network.predict(network.input_image_by_matrix()):
            print("{}: {}".format(obj[0], obj[1]))


if __name__ == "__main__":
    main()

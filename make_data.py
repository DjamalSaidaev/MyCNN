import numpy as np
import os  # для работы с файлами на диске
import random
import glob
import cv2 as cv
import time


random.seed(time.time())


def train_test_split(x, y, percent):
    num = int(percent * len(x))
    testx = x[:len(x) - num]
    testy = y[:len(y) - num]
    trainx = x[len(x) - num:]
    trainy = y[len(y) - num:]
    return np.array(trainx), np.array(testx), np.array(trainy), np.array(testy)


def read_data_sets():
    # берём пути к изображениям и рандомно перемешиваем
    data = []
    labels = []
    imagePaths = sorted(list(glob.glob("dataset/**/*.jpg", recursive=True)))
    random.shuffle(imagePaths)
    # цикл по изображениям
    for imagePath in imagePaths:
        # загружаем изображение, меняем размер на 8x8 пикселей (без учёта соотношения сторон)
        # добавляем в список
        # переводим изображение в черно-белое
        image = cv.imread(imagePath)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        data.append(gray)

        # извлекаем метку класса из пути к изображению и обновляем
        # список меток
        label = imagePath.split(os.path.sep)[-2]
        labels.append(int(label))
    # масштабируем интенсивности пикселей в диапазон[0, 1]
    data = np.array(data, dtype="float")
    data = data.reshape(data.shape[0], 8, 8)
    data /= 255.0
    labels = np.array(labels, dtype="int")

    # разбиваем данные на обучающую и тестовую выборки, используя 75% данных
    # для обучения и оставшиеся 25% для тестирования
    return train_test_split(data, labels, 0.25)
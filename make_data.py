# -*- coding: utf-8 -*-

import numpy as np
import os  # для работы с файлами на диске
import random
import glob
import cv2 as cv
import time


random.seed(time.time())


def train_test_split(x, y, percent):
    num = int(percent * len(x))
    trainx = x[:len(x) - num]
    trainy = y[:len(y) - num]
    testx = x[len(x) - num:]
    testy = y[len(y) - num:]
    to_file = (np.array(trainx), np.array(testx), np.array(trainy), np.array(testy))
    np.save("train_test_data.npy", {'data': to_file})
    return to_file


def make_train_test(num):
    data = []
    labels = []
    imagePaths = sorted(list(glob.glob("data/**/*.jpg", recursive=True)))
    # цикл по изображениям
    for i in range(num):
        # загружаем изображение, меняем размер на 8x8 пикселей (без учёта соотношения сторон)
        # добавляем в список
        # переводим изображение в черно-белое
        path = random.choice(imagePaths)
        image = cv.imread(path)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = gaussian_blur(gray)
        data.append(gray)

        # извлекаем метку класса из пути к изображению и обновляем
        # список меток
        label = path.split(os.path.sep)[-2]
        labels.append(int(label))
    # масштабируем интенсивности пикселей в диапазон[0, 1]
    data = np.array(data, dtype="float")
    data = data.reshape(data.shape[0], 8, 8)
    data /= 255.0
    labels = np.array(labels, dtype="int")

    # разбиваем данные на обучающую и тестовую выборки, используя 75% данных
    # для обучения и оставшиеся 25% для тестирования
    return train_test_split(data, labels, 0.25)

def gaussian_blur(img):
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

def load_image(str):
    image = cv.imread(str)
    image = cv.resize(image, (8, 8))
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = np.array(gray, dtype="float")
    gray /= 255.0
    # gaussian = gaussian_blur(gray)
    return gray


def read_data_sets():
    # берём пути к изображениям и рандомно перемешиваем
    data = []
    labels = []
    imagePaths = sorted(list(glob.glob("data/**/*.jpg", recursive=True)))
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

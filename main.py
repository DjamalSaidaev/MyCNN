import numpy as np


w_l = np.array([
    [1, 2],
    [3, 4]])

y_l = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [10, 11, 12, 13],
                [14, 15, 16, 17]])

conv_parameters = {
    'convolution': True,
    'stride': 1,
    'center_w_l': (0, 0)
}


maxpool_parameters = {
    'convolution': False,
    'stride': 2,
    'center_window': (0, 0),
    'window_shape': (2, 2)
}


def maxpool(input, params):
    indexes_a, indexes_b = create_indexes(size_axis=params['window_shape'], center_w_l=params['center_window'])
    stride = params['stride']
    # выходные матрицы будут расширяться по мере добавления новых элементов
    output = np.zeros((1, 1))  # матрица y_l после операции макспулинга
    # в зависимости от типа операции меняется основная формула функции
    if params['convolution']:
        g = 1  # операция конволюции
    else:
        g = -1  # операция корреляции
    # итерация по i и j входной матрицы y_l из предположения, что размерность выходной матрицы будет такой же
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            result = -np.inf
            element_exists = False
            for a in indexes_a:
                for b in indexes_b:
                    # проверка, чтобы значения индексов не выходили за границы
                    if (0 <= i*stride - g*a < input.shape[0]) and (0 <= j*stride - g*b < input.shape[1]):
                        if input[i*stride - g*a][j*stride - g*b] > result:
                            result = input[i*stride - g*a][j*stride - g*b]
                        element_exists = True
            # запись полученных результатов только в том случае, если для данных i и j были произведены вычисления
            if element_exists:
                if i >= output.shape[0]:
                    output = np.vstack((output, np.zeros(output.shape[1])))
                if j >= output.shape[1]:
                    # добавление столбца, если не существует
                    output = np.hstack((output, np.zeros((output.shape[0], 1))))
                output[i][j] = result
    return output


def convolution_feed_x_l(input, weights, params):
    indexes_a, indexes_b = create_indexes(size_axis=weights.shape, center_w_l=params['center_w_l'])
    stride = params['stride']
    # матрица выхода будет расширяться по мере добавления новых элементов
    output = np.zeros((1, 1))
    # в зависимости от типа операции меняется основная формула функции
    if params['convolution']:
        g = 1  # операция конволюции
    else:
        g = -1  # операция корреляции
    # итерация по i и j входной матрицы y_l_minus_1 из предположения,
    # что размерность выходной матрицы x_l будет такой же
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            # demo = np.zeros([y_l_minus_1.shape[0], y_l_minus_1.shape[1]]) # матрица для демонстрации конволюции
            result = 0
            element_exists = False
            for a in indexes_a:
                for b in indexes_b:
                    # проверка, чтобы значения индексов не выходили за границы
                    if (0 <= i*stride - g*a < input.shape[0]) and (0 <= j*stride - g*b < input.shape[1]):
                        result += input[i*stride - g*a][j*stride - g*b] * weights[indexes_a.index(a)][indexes_b.index(b)] # перевод индексов в "нормальные" для извлечения элементов из матрицы w_l
                        # demo[i*stride - g*a][j*stride - g*b] = w_l[indexes_a.index(a)][indexes_b.index(b)]
                        element_exists = True
            # запись полученных результатов только в том случае, если для данных i и j были произведены вычисления
            if element_exists:
                if i >= output.shape[0]:
                    # добавление строки, если не существует
                    output = np.vstack((output, np.zeros(output.shape[1])))
                if j >= output.shape[1]:
                    # добавление столбца, если не существует
                    output = np.hstack((output, np.zeros((output.shape[0], 1))))
                output[i][j] = result
                # вывод матрицы demo для отслеживания хода свертки
                # print('i=' + str(i) + '; j=' + str(j) + '\n', demo)
    return output


def create_axis_indexes(size_axis, center_w_l):
    coordinates = []
    for i in range(-center_w_l, size_axis-center_w_l):
        coordinates.append(i)
    return coordinates


def create_indexes(size_axis, center_w_l):
    # расчет координат на осях ядра свертки в зависимости от номера центрального элемента ядра
    coordinates_a = create_axis_indexes(size_axis=size_axis[0], center_w_l=center_w_l[0])
    coordinates_b = create_axis_indexes(size_axis=size_axis[1], center_w_l=center_w_l[1])
    return coordinates_a, coordinates_b


def main():
    temp = convolution_feed_x_l(y_l, w_l, conv_parameters)
    print("\nafter conv x_l\n", temp)
    out_maxpooling = maxpool(temp, maxpool_parameters)
    print("\nafter pooling x_l\n", out_maxpooling)


if __name__ == "__main__":
    main()

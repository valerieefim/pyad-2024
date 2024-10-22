import numpy as np


# задание 1
def matrix_multiplication(matrix_a, matrix_b):

    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0])
    cols_b = len(matrix_b[0])

    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Несоответствие размеров матриц.")

    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result

# задание 2
def functions(a_1, a_2):
    coeffs1 = list(map(int, a_1.split()))
    coeffs2 = list(map(int, a_2.split()))

    a1, b1, c1 = coeffs1
    a2, b2, c2 = coeffs2

    # на всякий случай проверим, не равны ли коэффициенты функций
    if coeffs1 == coeffs2:
        return None

    # решение уравнения f(x) = g(x) сводится к  a1*x^2 + b1*x + c1 = a2*x^2 + b2*x + c2
    # преобразуем ур-ния к виду (a1 - a2)x^2 + (b1 - b2)x + (c1 - c2) = 0
    A = a1 - a2
    B = b1 - b2
    C = c1 - c2

    disc = B**2 - 4*A*C

    if disc < 0:
        return []

    elif disc == 0:
        x = -B / (2*A)
        y = a1*x**2 + b1*x + c1
        return [(x, y)]

    else:
        x1 = (-B + np.sqrt(disc)) / (2*A)
        x2 = (-B - np.sqrt(disc)) / (2*A)
        y1 = a1*x1**2 + b1*x1 + c1
        y2 = a1*x2**2 + b1*x2 + c1
        return [(x1, y1), (x2, y2)]


# задание 3.1
def skew(x):
    n = len(x)
    mean_x = np.mean(x)
    std_x = np.std(x, ddof=1)
    skewness = (n / ((n - 1) * (n - 2))) * sum(((xi - mean_x) / std_x) ** 3 for xi in x)
    return round(skewness, 2)


# задание 3.2
def kurtosis(x):
    n = len(x)
    mean_x = np.mean(x)
    std_x = np.std(x, ddof=1)
    kurt = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * sum(((xi - mean_x) / std_x) ** 4 for xi in x)
    kurt -= (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return round(kurt, 2)
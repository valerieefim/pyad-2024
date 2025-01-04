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
    def count_extremum(coeffs):
        a, b, c = coeffs
        if a == 0:
            return None if b == 0 else -c / b
        else:
            return -b / (2 * a)

    def build_function(coeffs, x):
        a, b, c = coeffs
        return a * x ** 2 + b * x + c

    def count_mutual_solutions(a1, a2):
        a = a1[0] - a2[0]
        b = a1[1] - a2[1]
        c = a1[2] - a2[2]

        # рассмотрим несколько вариантов решения
        # 1 – все коэффициенты равны нулю, и решений бесконечно много
        if a == 0 and b == 0 and c == 0:
            return None

        # 2 – уравнение получилось линейным, поэтому решаем линейное уравнение
        if a == 0:
            if b == 0:
                return []
            else:
                x = -c / b
                return [(x, build_function(a1, x))]

        discriminant = b ** 2 - 4 * a * c

        # условия для 3-х видов дискриминантов
        if discriminant < 0:
            return []
        elif discriminant == 0:
            x = -b / (2 * a)
            return [(x, build_function(a1, x))]
        else:
            x1 = (-b + np.sqrt(discriminant)) / (2 * a)
            x2 = (-b - np.sqrt(discriminant)) / (2 * a)
            return [(x1, build_function(a1, x1)), (x2, build_function(a1, x2))]

    coeffs_1 = list(map(float, a_1.split()))
    coeffs_2 = list(map(float, a_2.split()))

    extremum_1 = count_extremum(coeffs_1)
    extremum_2 = count_extremum(coeffs_2)

    mutual_solutions = count_mutual_solutions(coeffs_1, coeffs_2)

    return mutual_solutions


# добавим вспомогательную функцию для расчета центрального момента заданного порядка.
def find_central_moment(x, order):
    return np.sum((x - np.mean(x)) ** order) / len(x)


# задание 3.1
def skew(x):
    m2 = find_central_moment(x, 2)
    m3 = find_central_moment(x, 3)
    sigma = np.sqrt(m2)
    skewness = m3 / sigma ** 3
    return round(skewness, 2)


# задание 3.2
def kurtosis(x):
    m2 = find_central_moment(x, 2)
    m4 = find_central_moment(x, 4)
    sigma = np.sqrt(m2)
    kurt = m4 / sigma ** 4 - 3
    return round(kurt, 2)

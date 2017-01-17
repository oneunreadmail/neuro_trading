"""WELL WELL WELL IT'S FAMOUS HARRY POTTER"""

import numpy as np
import time
# import matplotlib.pyplot as plt
# from report import Report
from timer import Timer, Timers


def sigma(x):  # Активационная функция. В данном случае, сигма.
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def deriv_sigma(x):  # Производная активационной функции. NB: аргумент — значение самой функции, не аргумента.
    return x * (1 - x)


class Perceptron():
    def __init__(self, structure):
        """

        :param structure:
         Лист вида [i, a, b, c, d... z, o] или [i, o], где
         i: количество элементов на входе перцептрона
         a..z: количество элементов в соответствующем промежуточном слое перцептрона
         o: количество элементов на выходе перцептрона (нейронов)

        """

        # input_range — число элементов на входе перцептрона в одном примере
        self.input_range = [structure[0]]
        # self.input = np.array(structure[0], ndmin=2)

        # output_range — число нейронов (выходов) у перцептрона
        self.output_range = [structure[-1]]
        self.output = np.array(structure[-1], ndmin=2)

        # middle_range — 0, если промежуточных слоев не требуется;
        #             число элементов в промежуточном слое, если он нужен один;
        #             массив вида [a, b, c], если нужно три слоя с числом элементов a, b и c, соответственно
        self.middle_range = structure[1:-1]
        self.middle = [np.array([], ndmin=2) for i in self.middle_range]

        self.values = [np.array([], ndmin=2) for i in structure]

        self.expand_range = structure  # Расширенный лист размерностей, типа [i, a, b, c, o].
        self.count = len(self.expand_range)-1  # Число итераций, для предыдущего примера — 4

        # Для каждой пары нам надо будет теперь задать матрицу перехода и вектор смещения
        self.matrix = []
        self.offset = []
        for i in range(self.count):  # На единицу меньше, так как матриц нужно на одну меньше
            self.matrix.append(np.array(2 * np.random.random((self.expand_range[i], self.expand_range[i+1])) - 1))
            self.offset.append(np.array(2 * np.random.random((1, self.expand_range[i+1])) - 1))

        self.alpha = 0.5  # Скорость обучения [0..1]
        self.beta = 0.  # Коэффициент учета инерции обучения
        self.total_error = -1

        self.iterations = 1000

        self.popt = self.pcov = 0  # Аппроксимация числа ошибок убывающей экспонентой

        self.mistakes = None
        self.E = None

        self.method = "sigma"

        # self.report = Report()

    def how_about_these(self, inpt):  # Считает ответ перцептрона на набор входных значений
        self.values[0] = np.array(inpt, ndmin=2)
        for i in range(0, self.count):
            self.values[i+1] = sigma(np.dot(self.values[i], self.matrix[i]) + self.offset[i])
        if self.method == "softmax":
            self.values[self.count] = softmax(np.dot(self.values[i], self.matrix[i]))
        self.output = self.values[-1]

    def learn_these(self, inpt, wanted_result, exit_by_count=True, print_progress=False):
        # Учит перцептрон на многих примерах
        self.mistakes = []
        self.E = []
        start = time.time()
        # self.input = np.array(input)

        error = [None] * (self.count + 1)

        t = Timers()

        matrix_delta_previous = [m * 0 for m in self.matrix]
        matrix_delta_current = [m * 0 for m in self.matrix]
        offset_delta_previous = [m * 0 for m in self.offset]
        offset_delta_current = [m * 0 for m in self.offset]

        if exit_by_count:
            for j in range(self.iterations):
                self.how_about_these(inpt) # 9.8

                self.example_range = self.values[0].shape[0]  # 0.1
                self.E.append(np.sum(np.absolute(
                    wanted_result - self.output)) / self.example_range / self.output_range[0])  # 1.4

                error[self.count] = np.array(deriv_sigma(self.output) * (wanted_result - self.output))  # 1.1
                if self.method == "softmax":  # 0.08
                    error[self.count] = np.array(wanted_result - self.output)
                # self.E.append(np.sum(np.abs(error[self.count]))) # 0.7
                for i in range(self.count - 1, 0, -1):  # 2.5
                    error[i] = deriv_sigma(self.values[i]) * np.dot(error[i + 1], self.matrix[i].T)
                for i in range(self.count):  # 10.0
                    matrix_delta_current[i] = np.tensordot(self.values[i], error[i + 1], axes=([0], [0]))
                    self.matrix[i] += self.alpha * matrix_delta_current[i] + \
                                      self.beta * matrix_delta_previous[i]
                    matrix_delta_previous[i] = matrix_delta_current[i]

                    offset_delta_current[i] = np.sum(error[i + 1], 0)
                    self.offset[i] += self.alpha * offset_delta_current[i] + \
                                      self.beta * offset_delta_previous[i]
                    offset_delta_previous[i] = offset_delta_current[i]

                if print_progress:  # 0.2
                    if j * 10 % self.iterations == 0:
                        if j == 0: print("0% ", sep='', end='', flush=True)
                        else: print(" ", j * 10 // self.iterations, "0% ", sep='', end='', flush=True)
                    elif j * 40 % self.iterations == 0:
                        print(".", sep='', end='', flush=True)

            if print_progress:
                print("100%", end = "\r", flush=True)
            # self.report.time = time.time() - start
            self.total_error = np.linalg.norm(wanted_result - self.output)
            self.E = np.array(self.E)
            # self.report.A, self.report.B, self.report.C, *rest = self.report.fit_exp(self.E)
            # self.report.spikiness = np.sum(np.abs(
            #     self.E - np.convolve(self.E, np.ones((100,))/100, mode="same"))) / self.iterations
            # print(self.report.spikiness)

    def how_close (self, input, output):
        o = np.array(output)
        # self.input = np.array(input)
        self.how_about_these(input)
        return round(np.linalg.norm(o - self.output)/o.size,3)

    def how_many_mistakes (self, input, output):
        o = np.array(output)
        # self.input = np.array(input)
        self.how_about_these(input)
        return int(np.sum(np.around(np.absolute(o - self.output))))


if __name__ == '__main__':
    # np.random.seed(43)

    N1 = Perceptron([3, 3, 2])
    inputs = [[0, 0, 0],
              [0, 1, 1],
              [0, 2, 1],
              [0, 0, 1],
              [0, 1, 0],
              [0, 2, 3],
              [1, 0, 1],
              [1, 1, 1],
              [1, 2, 1],
              [1, 0, 0],
              [1, 1, 2],
              [1, 2, 3],
              [1, 1, 0]]

    outputs = [[1, 0],
               [1, 0],
               [1, 1],
               [1, 0],
               [1, 0],
               [1, 1],
               [0, 0],
               [0, 0],
               [0, 1],
               [0, 0],
               [0, 0],
               [0, 1],
               [0, 0]]

    testinputs = [[0, 0, 3],
                  [0, 1, 0],
                  [0, 2, 1],
                  [1, 2, 0],
                  [1, 1, 3]]

    testoutputs = [[1, 0],
                   [1, 0],
                   [1, 1],
                   [0, 1],
                   [0, 0]]

    print("Ошибка на тестовых примерах до обучения:", N1.how_close(testinputs, testoutputs))
    print("Количество неправильных ответов:", N1.how_many_mistakes(testinputs, testoutputs))

    N1.iterations = 10000
    N1.learn_these(inputs, outputs, print_progress=True)

    print("\nОшибка на тестовых примерах после обучения:", N1.how_close(testinputs, testoutputs))
    print("Количество неправильных ответов:", N1.how_many_mistakes(testinputs, testoutputs))

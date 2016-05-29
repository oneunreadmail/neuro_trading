"""WELL WELL WELL IT'S FAMOUS HARRY POTTER"""

import numpy as np

def sigma(x): # Активационная функция. В данном случае, сигма.
   return 1/(1+np.exp(-x))

def deriv_sigma(x): # Производная активационной функции. NB: аргумент — значение самой функции, не аргумента.
   return x * (1-x)

class Perceptron:
   def __init__(self, inrange, midrange, outrange):
      self.input = np.array([])                                             # Входной массив
      self.i2m = np.array(2 * np.random.random((inrange, midrange)) - 1)    # input to middle — коэффициенты для расчета промежуточного слоя
      self.middle_offset = np.array(2 * np.random.random((1, midrange)) - 1) # Смещение для промежуточного
      self.middle = np.array([])                                            # Промежуточный слой
      self.m2o = np.array(2 * np.random.random((midrange,outrange)) - 1)    # middle to output — коэффициенты для расчета нейрона
      self.output_offset = np.array(2 * np.random.random((1, outrange)) - 1) # Смещение для выходного
      self.output = np.array([])                                            # Выходной массив или число, смотря сколько нейронов
      #self.offset = 0                              # Оффсет # Хует
      self.inrange = inrange                      # Число входов
      self.midrange = midrange                      # Число переменных в промежуточном слое
      self.outrange = outrange                      # Число нейронов
      self.alpha = 0.9                             # Скорость обучения [0..1]

   """ def how_about_this(self, input):                # Выдает единичный ответ перцептрона, какая-то хуйня пока
      self.make_friends()
      x = self.offset
      for i in range(0,len(self.middle)):
         x += self.middle[i] * self.matrix [i]
      r = 1 if x > 0 else 0
      return r """

   def how_about_these(self):        # Считает ответ перцептрона на набор входных значений
      self.middle = np.array(sigma(np.dot(self.input, self.i2m)),ndmin=2) #+ self.middle_offset
      self.output = np.array(sigma(np.dot(self.middle, self.m2o)),ndmin=2) #+ self.output_offset

   def learn_this(self, wanted_result):   # Учит перцептрон на одном примере
      self.how_about_these()
      out_error = np.array(deriv_sigma(self.output) * (wanted_result - self.output))
      delta_m2o = np.array(self.alpha * np.dot(self.middle.T, out_error))
      self.output_offset -= self.alpha * out_error
      middle_error = deriv_sigma(self.middle) * np.dot(out_error, self.m2o.T)
      delta_i2m = self.alpha * np.dot(self.input.T, middle_error)
      self.middle_offset -= self.alpha * middle_error

      self.m2o += delta_m2o
      self.i2m += delta_i2m
      return out_error.sum()

   def learn_these(self, wanted_result):   # Учит перцептрон на многих примерах
      self.how_about_these()
      out_error = np.array(deriv_sigma(self.output) * (wanted_result - self.output))
      #print("self.middle, out_error")
      #print(self.middle.shape, out_error.shape)
      delta_m2o = self.alpha * np.tensordot(self.middle, out_error, axes=([0],[0]))
      #print(self.output_offset)
      self.output_offset -= self.alpha * np.dot(np.ones((1,13)),out_error)
      middle_error = deriv_sigma(self.middle) * np.dot(out_error, self.m2o.T)
      delta_i2m = self.alpha * np.tensordot(self.input, middle_error, axes=([0,1],[0,1]))
      self.middle_offset -= self.alpha * np.dot(np.ones((1,13)),middle_error)

      self.m2o += delta_m2o
      self.i2m += delta_i2m
      return np.linalg.norm(out_error)



if __name__ == '__main__':
   np.random.seed(43)

   N1 = Perceptron(3, 3, 2)
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

   iterations = 10000

   N1.input = inputs
   N1.how_about_these()

   print(np.around(N1.output - outputs, 2))

   for j in range(iterations):
       if j * 10 % iterations == 0:
           print(" ", j * 10 // iterations, "0% ", sep='', end='', flush=True)
       elif j * 40 % iterations == 0:
           print(".", sep='', end='', flush=True)
       #print(iterations//10)
       """for i in range(len(inputs)):
           N1.input = np.array(inputs[i], ndmin=2)
           N1.learn_these(np.array(outputs[i], ndmin=2))"""

       N1.learn_these(outputs)





   print("")
   print("TAKTATTAK")

   print(np.around(N1.output - outputs, 2))


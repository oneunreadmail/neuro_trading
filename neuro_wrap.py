import pandas as pd
import numpy as np
import trading
import neuro

def empty_function(): pass

def slope(dataframe, column, a, b): # Источник, имя колонки, сколько значений до, сколько значений после
    up = down = 0.
    array = np.array(dataframe[column])
    for i in range(-a, b+1):
        #print(i, array[i])
        up += i * array[i]
        down += i * i
    #print()
    #print(array[-a], array[b])
    #print(array)
    #print(up/down)
    return up/down

def polynomial(dataframe, column, a, b):
    x = np.array(range(-a, b+1))
    y = np.array(dataframe[column][-a:b+1])
    z = np.polyfit(x, y, 1)
    return z

class Wrapper:

    lookback = 5                                # Сколько обрабатывать значений до текущего
    lookforward = 3                             # Сколько обрабатывать значений после текущего
    first_example = lookback                    # С какого начинаем, должен быть не меньше, чем lookback
    examples = 5                                # Количество чисел в примере, каждое в своем столбце


    def __init__(self, instrument, neuro):
        self.instrument = instrument
        self.neuro = neuro
        self.temp = None
        self.source = None
        self.result = None

    def source_rule(self, dataframe):
        r = list(dataframe["close"])
        return r - r[-1]

    def result_rule(self, dataframe):
        a = -2
        b = 2
        x = list(range(a, b+1))
        z = np.array(dataframe["close"])
        y = [z[x[i]] for i in range(len(x))]


        #print("\nx:", x, "\ny:", y, "\nz:", z)
        return np.polyfit(x, y, 1)
        #return [slope(dataframe, "close", 2, 2)]

    def make_source (self, proportions = None):

        source = []  # Будет инпутом
        result = []  # Будет аутпутом

        for i in range(0, self.examples):
            a = i + self.first_example - self.lookback
            b = a + self.lookback
            c = b + self.lookforward + 1
            source.append(self.source_rule(self.instrument.dataframe[a:b+1]))
            result.append(self.result_rule(self.instrument.dataframe[b:c].append(self.instrument.dataframe[a:b])))

        self.source = np.array(source)
        self.result = np.array(result)

        if (type(proportions) is list) and (proportions != []):
            p = [int(sum(proportions[0:i+1])/sum(proportions)*self.examples) for i in range(0, len(proportions))]
            r = np.split(self.source, p)
            print(r[0], "\n\n", r[1])


if __name__ == '__main__':
    N = neuro.Perceptron([10,10,2])
    RTS = trading.Instrument('data/SPFB.RTS_150101_160101.csv', 30)

    RTS.dataframe = pd.DataFrame({"close": list(range(50))}) # отладки ради

    W = Wrapper(RTS, N)
    W.make_source([60, 40])

    print()
    print(len(RTS.dataframe))
    #print("list time")
    #print(list(RTS.dataframe["close"]))
    print()
    print("\nsource:\n", W.source)
    print("\nresult:\n", W.result)


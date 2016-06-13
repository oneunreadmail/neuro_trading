import numpy as np
import pandas as pd

class Instrument:
    def __init__(self, filename, rows=None):
        self.filename = filename
        self.dataframe = pd.read_csv(filename, sep=";", nrows=rows, parse_dates=[[2, 3]])
        del self.dataframe['<PER>']
        del self.dataframe['<TICKER>']
        self.dataframe.columns = ("date_time", "open", "high", "low", "close", "volume")
        self.size = self.dataframe.shape[0]
        self.open = np.array(self.dataframe["close"]).T
        self.close = np.array(self.dataframe["close"]).T
        self.low = np.array(self.dataframe["low"]).T
        self.high = np.array(self.dataframe["high"]).T

class Broker:
    def __init__(self):
        self.deals = []
        self.dataframe = []
        self.inst = None
        self.deals = pd.DataFrame({"date_time": [], "type": [], "price": [], "link_number": []})
        self.absolute_comission = 0
        self.relative_comission = 0
        self.comission_per_contract = 0

        pass

    def simulate(self, instrument, action):
        """

        1 — buy
        0 — wait
        -1 — sell


        :param dataframe:
        :param action:
        :return:
        """
        matrix_deals = []
        prices = instrument.close
        dates = instrument.dataframe["date_time"].T
        money = 1
        if action[0] >= 0:
            matrix_deals.append([dates[0], 1, prices[0], 0, 1])
        else:
            matrix_deals.append([dates[0], -1, prices[0], 0, 1])
        size = min(len(prices), len(action))

        for i in range(1, size):
            if (action[i] * matrix_deals[-1][1] == -1) or (i == size - 1):
                k = (prices[i] / matrix_deals[-1][2])**matrix_deals[-1][1]
                matrix_deals.append([dates[i], -matrix_deals[-1][1], prices[i], i, k])
                money *= k


        self.deals = pd.DataFrame(matrix_deals,columns=("date_time", "type", "price", "link_number", "profit"))
        #self.deal_prices = deal_prices
        #self.outcome = round((money - 1) * 100, 4)

        self.outcome = matrix_deals[0][2]**matrix_deals[0][1] * matrix_deals[-1][2]**matrix_deals[-1][1]
        for i in range(1,len(matrix_deals)-1):
            self.outcome *= (matrix_deals[i][2]**matrix_deals[i][1])**2
        return money



if __name__ == '__main__':
    RTS = Instrument('data/SPFB.RTS_150101_160101.csv', 50)
    finam = Broker()
    finam.simulate(RTS, [0,1,0,-1,1,0,-1,-1,-1])
    #print(finam.deal_prices)
    print(finam.deals)
    print(finam.outcome)
    #print(finam.o2)
    #print()
    #print(RTS.close)
    #print(RTS.dataframe)

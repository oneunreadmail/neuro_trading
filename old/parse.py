import pandas as pd
import numpy as np


class Instrument:
    def __init__(self, filename, rows=None):
        self.filename = filename
        self.dataframe = pd.read_csv(filename, sep=";", encoding="cp1251", nrows=rows, parse_dates=[[2, 3]])
        del self.dataframe['<PER>']
        del self.dataframe['<TICKER>']
        self.dataframe.columns = ("date_time", "open", "high", "low", "close", "volume")
        self.size = self.dataframe.shape[0]
        self.open = np.array(self.dataframe["close"]).T
        self.close = np.array(self.dataframe["close"]).T
        self.low = np.array(self.dataframe["low"]).T
        self.high = np.array(self.dataframe["high"]).T


if __name__ == '__main__':
    RTS = Instrument('SPFB.RTS_150101_160101.csv', 50)
    #RTS.assign('SPFB.RTS_150101_160101.csv', 50)
    print(RTS.dataframe)
    ans = []

    print(len(RTS.dataframe["<CLOSE>"]))
    for i in range(0, len(RTS.dataframe["<CLOSE>"]) - 15):
        print(RTS.dataframe["<CLOSE>"][i + 11])
        # inp[i] = RTS.dataframe["<CLOSE>"][i,i+10]
        ans[i] = RTS.dataframe["<CLOSE>"][i + 11]
        print(ans)

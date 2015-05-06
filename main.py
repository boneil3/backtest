__author__ = 'brendan'

import pandas as pd
import numpy as np
from datetime import datetime as dt
from matplotlib import pyplot as plt


class Backtest():

    def __init__(self):
        self.prices = #get prices

    def strategy_SSA(self):
        """ Generate exposure from prices """

    def generate_pnl(self, exposure):
        delta_price = self.prices.pct_change()
        delta_portfolio = delta_price * exposure

        pnl = pd.DataFrame(np.ones(delta_portfolio.shape[0]), index=self.prices.index)
        for i in range(len(pnl.index) - 1):
            pnl.items[i+1] = (pnl.items[i] + 1) * delta_portfolio.items[i]
            
__author__ = 'brendan'

import pandas as pd
import numpy as np
from datetime import datetime as dt
from matplotlib import pyplot as plt
from main import Backtest

import random
import itertools
import time
import dateutil

import sys
sys.path.append('raw_data')

raw_data = pd.read_csv('raw_data/npr_history.csv')
data = pd.DataFrame(data=raw_data.iloc[13:, 1:].values, index=pd.to_datetime(raw_data.iloc[13:, 0]),
                    columns=[i for i in range(raw_data.shape[1] - 1)])

data.columns = ['Purchases of Domestic US', 'Sales of Domestic US', 'Net Domestic US Purchases', 'Net Private',
                'Net Private Treasuries', 'Net Private Agency Bonds', 'Net Private Corporate Bonds',
                'Net Private Equities', 'Net Official', 'Net Official Treasuries', 'Net Official Agency Bonds',
                'Net Official Corporate Bonds', 'Net Official Equities', 'Purchases of Foreign by US',
                'Sales of Foreign to US', 'Net Foreign Purchases by US', 'Net Foreign Bonds Purchases by US',
                'Net Foreign Equity Purchases by US', 'Net Transactions in Securities', 'Other Acq. of Securities',
                'Net Foreign Acq. of Securities', 'Change Foreign Owning Dollar ST Debt', 'Change Foreign Owning Bills',
                'Change in Foreign Private Owning Bills', 'Change in Foreign Official Owning Bills',
                'Change in Foreign Owning Other', 'Change in Foreign Private Owning Other',
                'Change in Foreign Official Owning Other', 'Change in Bank Net Dollar Liabilities', 'Net TIC Flow',
                'Net Private TIC Flow', 'Net Official TIC Flow']


data_MA3 = pd.rolling_mean(data, window=3, min_periods=1)
data_MA3.index.name = 'Date'

preds = []
data_names = ['Net Foreign Acq. of Securities', 'Change Foreign Owning Dollar ST Debt',
              'Change in Bank Net Dollar Liabilities', 'Net TIC Flow']
x = Backtest()
for i, name in enumerate(data_names):

    preds.append(pd.DataFrame(x.get_ssa_prediction(pd.DataFrame(data.ix[24:, name]), M=24), index=data.index))

fig, ax = plt.subplots(2, 2)
for i in range(len(preds)):
    if i == 0:
        preds[i].plot(ax=ax[0][0])
    if i == 1:
        preds[i].plot(ax=ax[1][0])
    if i == 2:
        preds[i].plot(ax=ax[0][1])
    if i == 3:
        preds[i].plot(ax=ax[1][1])


data_MA3['Net Foreign Acq. of Securities'].plot(ax=ax[0][0])
ax[0][0].set_title('Net Foreign Acq. of Securities')
data_MA3['Change Foreign Owning Dollar ST Debt'].plot(ax=ax[1][0])
ax[1][0].set_title('Change Foreign Owning Dollar ST Debt')
data_MA3['Change in Bank Net Dollar Liabilities'].plot(ax=ax[0][1])
ax[0][1].set_title('Change in Bank Net Dollar Liabilities')
data_MA3['Net TIC Flow'].plot(ax=ax[1][1])
ax[1][1].set_title('Net TIC Flow')
plt.tight_layout()
plt.show()
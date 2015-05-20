__author__ = 'brendan'

import main
import pandas as pd
import numpy as np
from datetime import datetime as dt
from matplotlib import pyplot as plt
import random
import itertools
import time
import dateutil
from datetime import timedelta

data = pd.read_csv('raw_data/MACRO_DATA.csv', index_col=0, parse_dates=True)
data.index.name = 'Date'
fix, ax = plt.subplots()
data['US_PCE'].dropna().plot(ax=ax, label='US PCE')
data['UK_CPI'].dropna().plot(ax=ax, label='UK CPI')
data['EUR_CPI'].dropna().plot(ax=ax, label='EUR CPI')
data['JP_CPI'].dropna().plot(ax=ax, label='JP CPI')
plt.legend()
ax.set_title('Inflation')

fig2, ax2 = plt.subplots()
data_uk_gdp_q = pd.DataFrame(pd.read_csv('raw_data/PGDP_CSDB_DS.csv', index_col=0, parse_dates=True).ix[:-8, 0]).astype(float) / 100.0
data_eur_gdp_q = pd.DataFrame(pd.read_csv('raw_data/data.csv', index_col=0, parse_dates=True, skiprows=5).ix[:, 1]).astype(float) / 100.0
data_uk_gdp_q.index = pd.DatetimeIndex(pd.date_range('1956-01-01', '2015-03-31', freq='QS'))
data_uk_gdp_q.index.name = 'Date'
data_eur_gdp_q.index.name = 'Date'
data_uk_gdp_q.columns = ['UK']
data_eur_gdp_q.columns = ['EUR']
print(data_uk_gdp_q)
print(data['US_GDP_Q'])
#data['US_GDP_Q'].dropna().pct_change(periods=4).plot(ax=ax2, label='US')
data_uk_gdp_q['UK'].plot(ax=ax2, label='UK')
data_eur_gdp_q.plot(ax=ax2, label='EUR')
#Xdata['JP_GDP_A'].dropna().pct_change(periods=1).plot(ax=ax2, label='JP')
ax2.set_title('Real GDP Growth')
plt.legend()

fig3, ax3 = plt.subplots()
data['US_ER'].dropna().plot(ax=ax3, label='US')
data['UK_ER'].dropna().plot(ax=ax3, label='UK')
data['GER_ER'].dropna().plot(ax=ax3, label='GER')
data['ITA_ER'].dropna().plot(ax=ax3, label='ITA')
data['FRA_ER'].dropna().plot(ax=ax3, label='FRA')
data['JP_ER'].dropna().plot(ax=ax3, label='JP')
ax3.set_title('Employment Ratio')
plt.legend()

fig4, ax4 = plt.subplots()
data['US_M2'].dropna().pct_change(periods=1).plot(ax=ax4, label='US')
data['UK_M2'].dropna().pct_change(periods=1).plot(ax=ax4, label='UK')
data['GER_M2'].dropna().pct_change(periods=1).plot(ax=ax4, label='GER')
data['ITA_M2'].dropna().pct_change(periods=1).plot(ax=ax4, label='ITA')
data['FRA_M2'].dropna().pct_change(periods=1).plot(ax=ax4, label='FRA')
data['JP_M2'].dropna().pct_change(periods=1).plot(ax=ax4, label='JP')
ax4.set_title('M2 Growth')
plt.legend()
plt.show()
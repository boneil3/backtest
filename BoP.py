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

cols = ['BoP FA Net', 'BoP FA OI Net', 'BoP FA PI Net', 'CA % GDP']
raw_data = pd.read_csv('raw_data/BoP_UK.csv', index_col=0, parse_dates=True)
data = pd.DataFrame(raw_data.iloc[:240, :4].fillna(0)).astype(float)
data.columns = cols
data.index = pd.date_range('1955-01-01', '2014-12-31', freq='Q')

raw_eur = pd.read_csv('raw_data/EUR_CA.csv', index_col=0, parse_dates=True)
raw_eur = raw_eur[::-1]
raw_eur.index = pd.date_range('1999-01-01', '2015-03-01', freq='M')
raw_eur.index.name = 'Date'
raw_eur = raw_eur.resample('Q', how='sum')

data_eur_gdp_q = pd.read_csv('raw_data/MACRO_DATA.csv', index_col=0, parse_dates=True)['EUR_GDP_Q'].dropna()
data_eur_gdp_q.columns = ['EUR_GDP_Q']
data_eur_gdp_q.index.name = 'Date'
data_eur_gdp_q = data_eur_gdp_q.loc['1999-03-31':]
end_gdp = pd.DataFrame(data=[data_eur_gdp_q.iloc[-1], data_eur_gdp_q.iloc[-1],
                             data_eur_gdp_q.iloc[-1], data_eur_gdp_q.iloc[-1]],
                       index=pd.date_range('2014-06-30', '2015-03-31', freq='Q'))

eur_gdp = pd.concat([data_eur_gdp_q, end_gdp])
eur_gdp.columns = ['EUR_CA']

eur_ca = raw_eur.div(eur_gdp)

eur_ca.columns = ['EUR CA']
uk_ca = data['CA % GDP'] / 100.0
uk_ca.columns = ['UK CA']

uk_fa = pd.DataFrame(data.iloc[:, :3])
uk_gdp = pd.read_csv('raw_data/MACRO_DATA.csv', index_col=0, parse_dates=True)['UK_GDP_Q'].dropna()

uk_gdp_final = pd.concat([uk_gdp, pd.DataFrame(data=[uk_gdp.iloc[-1], uk_gdp.iloc[-1]],
                                               index=pd.date_range('2014-09-01', '2014-12-31', freq='Q'))])

uk_fa_gdp = pd.DataFrame(index=uk_gdp_final.index)
uk_fa_gdp['UK FA Net'] = uk_fa['BoP FA Net'] / uk_gdp_final
uk_fa_gdp['UK FA OI'] = uk_fa['BoP FA OI Net'] / uk_gdp_final
uk_fa_gdp['UK FA PI'] = uk_fa['BoP FA PI Net'] / uk_gdp_final

print(eur_gdp)
eur_fa = pd.read_csv('raw_data/EUR_FA.csv', index_col=0, header=0, parse_dates=True).dropna().astype(float)
eur_fa = eur_fa.iloc[::-1]
print(eur_fa)
eur_fa.index = pd.date_range('2009-01-01', '2015-02-28', freq='M')
eur_fa = eur_fa.resample('Q', how='sum')
print(eur_fa)
eur_fa_gdp = pd.DataFrame(index=eur_gdp.index)
eur_fa_gdp['EUR FA Net'] = eur_fa['EUR FA Net'] / eur_gdp['EUR_CA'].loc['2009-03-31':]
eur_fa_gdp['EUR FA OI'] = eur_fa['EUR FA OI'] / eur_gdp['EUR_CA'].loc['2009-03-31':]
eur_fa_gdp['EUR FA PI'] = eur_fa['EUR FA PI'] / eur_gdp['EUR_CA'].loc['2009-03-31':]
print(eur_fa_gdp)
fig, ax = plt.subplots()
uk_ca.plot(ax=ax, label='UK CA')
eur_ca.plot(ax=ax, label='EUR CA')
ax.set_title('Current Account %GDP')
plt.legend()

uk_fa_gdp_4q = pd.rolling_mean(uk_fa_gdp, window=4)
fig2, ax2 = plt.subplots()
uk_fa_gdp_4q.plot(ax=ax2)
#eur_fa_gdp.plot(ax=ax2)
plt.legend()
ax2.set_title('UK Financial Account % GDP (4Q Avg.)')
#plt.show()

dates = pd.DataFrame(index=pd.date_range('1960-03-31', '2015-01-01', freq='Q'))
print(dates)
dates.to_csv('raw_data/US_BoP.csv')
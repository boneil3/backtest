__author__ = 'brendan'
import Quandl
import pandas as pd
secs = ['EURUSD', 'GBPUSD', 'EURGBP', 'AUDUSD', 'USDMXN', 'USDINR', 'USDBRL', 'USDCAD', 'USDZAR']

datas = []
for i, sec in enumerate(secs):
    data = pd.DataFrame(Quandl.get('CURRFX/' + sec, authtoken='ZoAeCkDnkL4oFQs1z2_u')['Rate'])
    data = data.loc['2000-01-03':]
    data.columns = [sec]
    print(data)
    if i == 0:
        ret_data = data
        continue
    ret_data = ret_data.join(data, how='left')
ret_data.to_csv('CCY_2000.csv')
__author__ = 'brendan'
import Quandl
import pandas as pd
#secs = ['EURUSD', 'GBPUSD', 'EURGBP', 'AUDUSD', 'USDMXN', 'USDINR', 'USDBRL', 'USDCAD', 'USDZAR']

#datas = []
#for i, sec in enumerate(secs):
#    data = pd.DataFrame(Quandl.get('CURRFX/' + sec, authtoken='ZoAeCkDnkL4oFQs1z2_u')['Rate'])
#    data = data.loc['2000-01-03':]
#    data.columns = [sec]
#    print(data)
#    if i == 0:
#        ret_data = data
#        continue
#    ret_data = ret_data.join(data, how='left')
#ret_data.to_csv('CCY_2000.csv')
authtoken = 'ZoAeCkDnkL4oFQs1z2_u'
'''da = []
data_names = ['US_PCE', 'UK_CPI', 'EUR_CPI', 'JP_CPI', 'US_GDP_Q', 'UK_GDP_Q', 'EUR_GDP_Q', 'JP_GDP_A', 'US_CP',
              'US_ER', 'UK_ER', 'ITA_ER', 'FRA_ER', 'GER_ER', 'JP_ER', 'US_M2', 'UK_M2', 'GER_M2', 'ITA_M2',
              'FRA_M2', 'JP_M2', 'US_CA', 'UK_CA', 'GER_CA', 'ITA_CA', 'FRA_CA', 'JP_CA', 'US_LF', 'UK_LF', 'GER_LF',
              'ITA_LF', 'FRA_LF', 'JP_LF', 'US_POP', 'UK_POP', 'GER_POP', 'ITA_POP', 'FRA_POP', 'JP_POP', 'US_POP_65',
              'UK_POP_65', 'GER_POP_65', 'ITA_POP_65', 'FRA_POP_65', 'JP_POP_65']
# Inflation Monthly
US_PCE = pd.DataFrame(Quandl.get('FRED/PCETRIM6M680SFRBDAL', authtoken=authtoken))
UK_CPI = pd.DataFrame(Quandl.get('UKONS/MM23_D7G7_M', authtoken=authtoken))
EUR_CPI = pd.DataFrame(Quandl.get('RATEINF/INFLATION_EUR', authtoken=authtoken))
JP_CPI = pd.DataFrame(Quandl.get('RATEINF/INFLATION_JPN', authtoken=authtoken))
da.extend([US_PCE, UK_CPI, EUR_CPI, JP_CPI])
# GDP Aggregate
US_GDP_Q = pd.DataFrame(Quandl.get('FRED/GDP', authtoken=authtoken))
UK_GDP_Q = pd.DataFrame(Quandl.get('UKONS/QNA_BKTL_Q', authtoken=authtoken))
EUR_GDP_Q = pd.DataFrame(Quandl.get('ECB/RTD_Q_S0_S_G_GDPM_TO_U_E', authtoken=authtoken))
JP_GDP_A = pd.DataFrame(Quandl.get('ODA/JPN_NGDP', authtoken=authtoken))  # Annual
da.extend([US_GDP_Q, UK_GDP_Q, EUR_GDP_Q, JP_GDP_A])

# Wage growth - YoY CANNOT FIND

# Corporate Profits Unadjusted
US_CP = pd.DataFrame(Quandl.get('FRED/CP', authtoken=authtoken))
da.append(US_CP)
# Employment ratio
US_ER = pd.DataFrame(Quandl.get('FRED/EMRATIO', authtoken=authtoken))
UK_ER = pd.DataFrame(Quandl.get('FRED/GBREPRNA', authtoken=authtoken))
ITALY_ER = pd.DataFrame(Quandl.get('FRED/ITAEPRNA', authtoken=authtoken))
FRANCE_ER = pd.DataFrame(Quandl.get('FRED/FRAEPRNA', authtoken=authtoken))
GER_ER = pd.DataFrame(Quandl.get('FRED/DEUEPRNA', authtoken=authtoken))
JAPAN_ER = pd.DataFrame(Quandl.get('FRED/JPNEPRNA', authtoken=authtoken))
da.extend([US_ER, UK_ER, ITALY_ER, FRANCE_ER, GER_ER, JAPAN_ER])
# M2
US_M2 = pd.DataFrame(Quandl.get('WORLDBANK/USA_FM_LBL_MQMY_GD_ZS', authtoken=authtoken))
UK_M2 = pd.DataFrame(Quandl.get('WORLDBANK/GBR_FM_LBL_MQMY_GD_ZS', authtoken=authtoken))
GER_M2 = pd.DataFrame(Quandl.get('WORLDBANK/DEU_FM_LBL_MQMY_GD_ZS', authtoken=authtoken))
ITA_M2 = pd.DataFrame(Quandl.get('WORLDBANK/ITA_FM_LBL_MQMY_GD_ZS', authtoken=authtoken))
FRA_M2 = pd.DataFrame(Quandl.get('WORLDBANK/FRA_FM_LBL_MQMY_GD_ZS', authtoken=authtoken))
JP_M2 = pd.DataFrame(Quandl.get('WORLDBANK/JPN_FM_LBL_MQMY_GD_ZS', authtoken=authtoken))
da.extend([US_M2, UK_M2, GER_M2, ITA_M2, FRA_M2, JP_M2])
# Current Account as % of GDP
US_CA = pd.DataFrame(Quandl.get('ODA/USA_BCA_NGDPD', authtoken=authtoken))
UK_CA = pd.DataFrame(Quandl.get('ODA/GBR_BCA_NGDPD', authtoken=authtoken))
GER_CA = pd.DataFrame(Quandl.get('ODA/DEU_BCA_NGDPD', authtoken=authtoken))
ITA_CA = pd.DataFrame(Quandl.get('ODA/ITA_BCA_NGDPD', authtoken=authtoken))
FRA_CA = pd.DataFrame(Quandl.get('ODA/FRA_BCA_NGDPD', authtoken=authtoken))
JP_CA = pd.DataFrame(Quandl.get('ODA/JPN_BCA_NGDPD', authtoken=authtoken))
da.extend([US_CA, UK_CA, GER_CA, ITA_CA, FRA_CA, JP_CA])
# Labor Force
US_LF = pd.DataFrame(Quandl.get('ODA/USA_LE', authtoken=authtoken))
UK_LF = pd.DataFrame(Quandl.get('ODA/GBR_LE', authtoken=authtoken))
GER_LF = pd.DataFrame(Quandl.get('ODA/DEU_LE', authtoken=authtoken))
ITA_LF = pd.DataFrame(Quandl.get('ODA/ITA_LE', authtoken=authtoken))
FRA_LF = pd.DataFrame(Quandl.get('ODA/FRA_LE', authtoken=authtoken))
JP_LF = pd.DataFrame(Quandl.get('ODA/JPN_LE', authtoken=authtoken))
da.extend([US_LF, UK_LF, GER_LF, ITA_LF, FRA_LF, JP_LF])
# Population
US_POP = pd.DataFrame(Quandl.get('ODA/USA_LP', authtoken=authtoken))
UK_POP = pd.DataFrame(Quandl.get('ODA/USA_LP', authtoken=authtoken))
GER_POP = pd.DataFrame(Quandl.get('ODA/USA_LP', authtoken=authtoken))
ITA_POP = pd.DataFrame(Quandl.get('ODA/USA_LP', authtoken=authtoken))
FRA_POP = pd.DataFrame(Quandl.get('ODA/USA_LP', authtoken=authtoken))
JP_POP = pd.DataFrame(Quandl.get('ODA/USA_LP', authtoken=authtoken))
da.extend([US_POP, UK_POP, GER_POP, ITA_POP, FRA_POP, JP_POP])
US_POP_65 = pd.DataFrame(Quandl.get('WORLDBANK/USA_SP_POP_65UP_TO_ZS', authtoken=authtoken))
UK_POP_65 = pd.DataFrame(Quandl.get('WORLDBANK/GBR_SP_POP_65UP_TO_ZS', authtoken=authtoken))
GER_POP_65 = pd.DataFrame(Quandl.get('WORLDBANK/DEU_SP_POP_65UP_TO_ZS', authtoken=authtoken))
ITA_POP_65 = pd.DataFrame(Quandl.get('WORLDBANK/ITA_SP_POP_65UP_TO_ZS', authtoken=authtoken))
FRA_POP_65 = pd.DataFrame(Quandl.get('WORLDBANK/FRA_SP_POP_65UP_TO_ZS', authtoken=authtoken))
JP_POP_65 = pd.DataFrame(Quandl.get('WORLDBANK/JPN_SP_POP_65UP_TO_ZS', authtoken=authtoken))

da.extend([US_POP_65, UK_POP_65, GER_POP_65, ITA_POP_65, FRA_POP_65, JP_POP_65])

data = da[0]
data.columns = ['US_PCE']
for i, df in enumerate(da):
    if i != 0:
        df.columns = [data_names[i]]
        data = data.join(df, how='outer')
print(data)
data.to_csv('MACRO_DATA.csv')
'''
US_EQ = pd.DataFrame(Quandl.get('CHRIS/CME_ES1', authtoken=authtoken)['Settle'])
UK_EQ = pd.DataFrame(Quandl.get('YAHOO/INDEX_FTSE', authtoken=authtoken)['Close'])
GER_EQ = pd.DataFrame(Quandl.get('YAHOO/INDEX_GDAXI', authtoken=authtoken)['Close'])
ITA_EQ = pd.DataFrame(Quandl.get('YAHOO/INDEX_FTSEMIB_MI', authtoken=authtoken)['Close'])
FRA_EQ = pd.DataFrame(Quandl.get('YAHOO/INDEX_FCHI', authtoken=authtoken)['Close'])
JP_EQ = pd.DataFrame(Quandl.get('NIKKEI/INDEX', authtoken=authtoken)['Close Price'])

CL = pd.DataFrame(Quandl.get('CHRIS/CME_CL1', authtoken=authtoken)['Settle'])
TY = pd.DataFrame(Quandl.get('CHRIS/CME_TY1', authtoken=authtoken)['Settle'])
TU = pd.DataFrame(Quandl.get('CHRIS/CME_TU1', authtoken=authtoken)['Settle'])

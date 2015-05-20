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
import Quandl


authtoken = 'ZoAeCkDnkL4oFQs1z2_u'
data = pd.read_csv('raw_data/COT_CRUDE.csv', index_col=0, parse_dates=True)
data_cop = pd.read_csv('raw_data/COT_COPPER.csv', index_col=0, parse_dates=True)
fig, ax = plt.subplots()
data.plot(ax=ax)
ax.set_title('Commitments of Traders in WTI')

fig2, ax2 = plt.subplots()
data_cop.plot(ax=ax2)
ax2.set_title('Commitments of Traders in Copper Grade #1')
plt.show()


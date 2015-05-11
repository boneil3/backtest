__author__ = 'brendan'

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

pData = np.array([11.50, 10.85, 11.15, 10.7, 11.1, 11.6, 11.85, 11.1, 10.95, 10.5, 10.9])
cData = np.array([12.7, 12.75, 12.5, 12.55])

pDelta = 12.50 - pData
cDelta = 13.0 - cData

fig, ax = plt.subplots()
plt.boxplot(pDelta)
plt.boxplot(cDelta)
plt.title('2014 AFC Championship Game: Pre-Game vs Halftime')
plt.ylabel('PSI')
plt.ylim([0, 2.2])
plt.show()
__author__ = 'brendan'

import pandas as pd
import numpy as np
from datetime import datetime as dt
from matplotlib import pyplot as plt
import random
import itertools
import time
import dateutil


class Backtest():
    def __init__(self):
        self.prices = pd.read_csv('CCY_daily.csv', index_col=0, parse_dates=True)

    def get_ssa_prediction(self, es, M=200):
        """ Generate prediction from prices """

        # ES OF TYPE DATAFRAME PANDAS
        d = 20
        bewl = False
        N = es.shape[0]
        timeStart = time.clock()
        if int(N / 2) == N / 2:
            es = es[1:]
            N -= 1
        L = int(N / 2) + 1
        K = N - L + 1
        # N-K+1 = L
        #L = N/2
        #N/2 = N-K+1
        #K = N/2+1 = L+1
        #2K -N - 1= 1

        esix = es.iloc
        X = pd.DataFrame([esix[i:i + L, 0].tolist() for i in range(K)])
        #for i in range(K):
        #XTix[i] = esix[i:i+L]

        XT = X.T

        #H = np.mat(X)*np.mat(XT)

        #H = X.dot(X.T)
        #sigma,V = scipy.linalg.eigh(H)
        U, sigma, V = np.linalg.svd(X, full_matrices=False)
        U = pd.DataFrame(U)
        V = pd.DataFrame(V)
        UT = U.T


        #Potential Pairs
        #### 0-14 ###
        #periodogram analysis

        Xtilde = [np.array(sigma[i] * np.outer(U.ix[:, i].values, V.ix[i].values))
                  for i in range(d)]


        ##create X1 X2 X3 GROUPED
        ##possible pairs on eigenvalue magnitude analysis



        XX = np.zeros((3, L, L))

        XX[0] = np.sum(Xtilde[0:3], axis=0, out=np.zeros((L, L)))
        XX[1] = np.sum(Xtilde[3:5], axis=0, out=np.zeros((L, L)))
        XX[2] = np.sum(Xtilde[5:], axis=0, out=np.zeros((L, L)))

        XXsum0 = [[1.0 / (k + 1) * (np.sum([XX[j, i, k - i] for i in range(k + 1)]))
                   for k in range(L - 1)] for j in range(3)]
        XXsum1 = [[1.0 / L * (np.sum([XX[j, i, k - i] for i in range(L)]))
                   for k in range(L - 1, K)] for j in range(3)]
        XXsum2 = [[1.0 / (N - k) * (np.sum([XX[j, i - 1, k - i + 1] for i in range(k - K + 2, N - K + 2)]))
                   for k in range(K, N)] for j in range(3)]
        #k = L-1 XX[0,L-1] [1,L-2], [
        #k = K -> 1/(L-1)*np.sum([XX[1,L-1]
        #k=N -> sum(XX[
        #N-1-
        #N-K=L-1
        #K-N-K-1 = 2K-N-1 = 0
        #K-1=L

        g0 = np.concatenate((XXsum0[0], XXsum1[0], XXsum2[0]))
        g1 = np.concatenate((XXsum0[1], XXsum1[1], XXsum2[1]))
        g2 = np.concatenate((XXsum0[2], XXsum1[2], XXsum2[2]))
        g = g0 + g1 + g2

        #k = N-1
        #N-K+1,N-K+1
        # L-1,L
        #K-N+K-1
        #2K-N-1 = 2(N-L+1)-N-1 = N - 2L + 1 = 0

        gPrime = g

        Uiloc = U.iloc

        pi = np.zeros(d)
        pi = [Uiloc[-1, i] for i in range(d)]
        #for i in range(d):
        #   pi[i] = U.ix[-1,i]

        vS = np.linalg.norm(pi, ord=2, axis=0) ** 2

        Rp = np.zeros((d, L - 1))

        Rp = [pi[i] * (Uiloc[:L - 1, i]) for i in range(d)]

        R = 1 / (1 - vS) * np.sum(Rp, axis=0)
        R = R[::-1]
        #How many predictions? M

        g2 = np.zeros(N + M)

        g2[:N] = gPrime
        if bewl:
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
            axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
            for i, ax in enumerate(axes):
                ax.plot(U.ix[:, i])
            plt.show()
        for i in range(N, N + M):
            g2[i] = np.sum([R[j] * g2[i - j - 1] for j in range(L - 1)])

        print("SSALoop " + str(es.index[-1]) + ": " + str(time.clock() - timeStart) + " sec")

        return g2

    def generate_exposure(self, sec_num):
        """
        :return: exposure to calculate PnL
        """

        prices = self.prices
        N = prices.shape[0]
        m = 600  # number of days out of sample, N-m = days in sample
        min_delta = .01  # min prediction move to change exposure
        if not isinstance(prices, pd.DataFrame):
            raise AssertionError('prices must be a DataFrame')

        # array of dataframes of predictions
        oo_sample = range(N - m, N - 1, 30)
        oo_sample_idx = [prices.index[i] for i in oo_sample]

        predictions = [self.get_ssa_prediction(pd.DataFrame(prices.iloc[:i-1, sec_num]), M=m - j * 30 + 1) for j, i in
                       enumerate(oo_sample)]
        pred_iter = zip(oo_sample_idx, predictions)

        pred_dfs = []
        for idx, pred in pred_iter:
            ret_pred = pd.DataFrame(pred, index=prices.index, columns=[str(idx)])
            pred_dfs.append(ret_pred)

        exposure = np.zeros(N)
        for i, idx in enumerate(prices.index):
            if i == 0:
                pass
            elif idx in oo_sample_idx:
                inter_pred = None
                for pred in pred_dfs:
                    if pred.columns == [str(idx)]:
                        inter_pred = pred
                p_data = inter_pred.iloc[i:, 0].values
                a = np.diff(np.sign(np.diff(p_data))).nonzero()[0] + 1

                pred_0 = inter_pred.iloc[i, 0]
                val_0 = prices.iloc[i, sec_num]
                if a.size >= 2 and abs((pred_0 - val_0) / val_0) < .01:
                    extrema_1 = inter_pred.iloc[i + a[0], 0]
                    extrema_2 = inter_pred.iloc[i + a[1], 0]
                    if (extrema_1 - pred_0) / pred_0 > min_delta and (extrema_2 - pred_0) / pred_0 > min_delta:
                        exposure[i] = 1
                    elif (extrema_1 - pred_0) / pred_0 < -min_delta and (extrema_2 - pred_0) / pred_0 < -min_delta:
                        exposure[i] = -1
                    else:
                        exposure[i] = exposure[i - 1]
                else:
                    exposure[i] = exposure[i - 1]
            else:
                exposure[i] = exposure[i - 1]

            # Stop Loss Check
            # First find last local extremum
            if i > N-m:
                if (exposure[i-1] > 0 and (self.prices.iloc[i-1, sec_num] - val_0) / val_0 < -.015) or \
                        (exposure[i-1] < 0 and (self.prices.iloc[i-1, sec_num] - val_0) / val_0 > .015):
                    exposure[i] = 0

        return exposure

    def generate_pnl(self, exposure, sec_num):
        delta_price = self.prices.pct_change()
        delta_portfolio = pd.DataFrame(delta_price.iloc[:, sec_num].values * exposure, index=self.prices.index)

        pnl = pd.DataFrame(data=np.ones_like(exposure), index=self.prices.index, columns=['Value'])
        for i, val in enumerate(delta_portfolio.iloc[:, 0].values):
            if i == 0:
                pass
            else:
                pnl.iloc[i] = pnl.iloc[i - 1] * (1.0 + val)
        return pnl


x = Backtest()
fig, ax = plt.subplots()
pnls = []
for sec_num in range(len(x.prices.columns)):
    if x.prices.columns[sec_num] == 'USDJPY':
        continue
    print(x.prices.columns[sec_num])
    exp = x.generate_exposure(sec_num)
    pnl = x.generate_pnl(exp, sec_num)
    pnl.columns = [str(x.prices.columns[sec_num])]
    pnls.append(pnl)
    pnl.plot(ax=ax)
    pnl.columns = ['Value']

total_pnl = pnls[0]
for i in range(1, len(pnls)):
    total_pnl = total_pnl + pnls[i]
total_pnl = total_pnl / float(len(pnls))
fig2, ax2 = plt.subplots()
total_pnl.plot(ax=ax2)
plt.show()

# TODO: Stop losses on bad predictions
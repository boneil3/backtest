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
        self.prices =  # get prices

    def get_ssa_prediction(self):
        """ Generate prediction from prices """

        # ES OF TYPE DATAFRAME PANDAS
        es = self.prices
        M = 100
        d = 20
        bewl = False
        N = es.shape[0]
        # print(N)
        timeStart = time.clock()
        if int(N / 2) == N / 2:
            es = es[1:]
            N -= 1
        L = int(N / 2) + 1
        K = N - L + 1

        #N-K+1 = L
        #L = N/2
        #N/2 = N-K+1
        #K = N/2+1 = L+1
        #2K -N - 1= 1
        XT = pd.DataFrame(np.zeros((K, L)))
        a = np.zeros(L)

        esix = es.ix

        X = pd.DataFrame([esix[i:i + L].values for i in range(K)])
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

        VT = V.T
        VTix = VT.ix
        UTix = UT.ix


        #Potential Pairs
        #### 0-14 ###
        #periodogram analysis

        Xtilde = np.zeros((d, L, L))
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
        ##    for i in range(d):
        ##        Rp[i] = pi[i]*P[:L-1,i]
        R = np.zeros(L - 1)
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
    

    def generate_pnl(self, exposure):
        delta_price = self.prices.pct_change()
        delta_portfolio = delta_price * exposure

        pnl = pd.DataFrame(np.ones(delta_portfolio.shape[0]), index=self.prices.index)
        for i in range(len(pnl.index) - 1):
            pnl.items[i + 1] = (pnl.items[i] + 1) * delta_portfolio.items[i]
            
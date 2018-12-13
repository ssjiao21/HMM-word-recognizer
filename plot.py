#!/usr/bin/env python

from numpy import genfromtxt
import pickle
import matplotlib.pyplot as plt


dataProbPr = genfromtxt('dataProb_pri.csv',delimiter=',')
dataProbCt = genfromtxt('dataProb_cntra.csv',delimiter=',')

### plot average data prob of primary model ###
fig1 = plt.figure()
plt.plot(range(1,len(dataProbPr)+1),dataProbPr)
plt.xlabel('number of iterations')
plt.ylabel('average log-prob of data')
plt.grid()
plt.savefig('train_pri.png')
plt.close(fig1)

### plot accuracy on held-out data of contrastive model ###
fig2 = plt.figure()
plt.plot(range(1,len(dataProbCt)+1),dataProbCt)
plt.xlabel('number of iterations')
plt.ylabel('average log-prob of data')
plt.grid()
plt.savefig('train_cntra.png')
plt.close(fig2)


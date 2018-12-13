#!/usr/bin/env python

import numpy as np
import math
import pickle
from copy import deepcopy

##### ----------------------------- Function ----------------------------- #####
def HMM_word(eHMM,sHMM):
	wHMM = {}
	for wd in baseFm:
		trsi = sHMM[0]
		emis = sHMM[1][:]
		for i in range(len(baseFm[wd])):
			trsi = np.concatenate((trsi,eHMM[baseFm[wd][i]][0]))
		for i in range(len(baseFm[wd])):
			for em in eHMM[baseFm[wd][i]][1]:
				emis.append(em)
		# print trsi.shape
		# print len(emis)
		trsi = np.concatenate((trsi,sHMM[0]))
		for each in sHMM[1]:
			emis.append(each)
		# print trsi
		# print len(emis)
		wHMM[wd] = [trsi,emis[:]]
	return wHMM
##### -------------------------------------------------------------------- #####
print '---------------- START -----------------'

file = open('clsp.lblnames','r')
lblName = []
for line in file:
	lblName.append(line.rstrip('\n'))
lblName.pop(0)
file.close()
# print lblName

file = open('clsp.devlbls','r')
tesLbls = []
for line in file:
	tmp = line.rstrip('\n').split(' ')
	if '' in tmp:
		tmp.remove('')
	tesLbls.append(tmp)
	# print len(tmp)
tesLbls.pop(0)
file.close()
# print tesLbls

##### Load training parameters #####
eleHMM1 = np.load('eleHMM_cntra.npy')
eleHMM = deepcopy(eleHMM1[()])
# print 'AA:\n',eleHMM['AA']
with open('silHMM_cntra','rb') as f:
    silHMM = pickle.load(f)
# print 'Silence:\n',silHMM

# ------------------------------------------------------------
# ##### Create elementary models (1) #####
# eleHMM = {}
# for lbl in lblName:
# 	tmp = []
# 	tmp.append(np.array([0.8,0.1,0.1]))	
# 	tmp2 = np.ones(256)*0.5/255
# 	tmp2[lblName.index(lbl)] = 0.5
# 	tmp.append([tmp2]*2+[np.zeros(256)])
# 	eleHMM[lbl] = tmp[:]
# # print 'AA:\n',eleHMM['AA']

# ##### Create the silence model (2) #####
# silHMM = []
# silHMM.append(np.ones(12)/2)
# tmp2 = np.ones(256)/256
# silHMM.append([tmp2]*9+[np.zeros(256)]*3)
# # print 'Silence:\n',silHMM
# ------------------------------------------------------------

##### Load fenonic baseforms #####
baseFm1 = np.load('baseform_cntra.npy')
baseFm = deepcopy(baseFm1[()])
# print baseFm.keys()
# print 'money:',baseFm['money']
# print 'enjoy:',baseFm['enjoy']

##### Form word HMMs (4) #####
wordHMM = HMM_word(eleHMM,silHMM)
# print wordHMM.keys()
# print 'enjoy:',wordHMM['enjoy']
# print wordHMM['enjoy'][0].shape
# print len(wordHMM['enjoy'][1])

##### Test #####
dataProb = np.zeros([len(tesLbls),len(baseFm.keys())])
# print dataProb.shape
idy = []
conf = []

ut = 0
for ii in range(len(tesLbls)):
	ut = ut+1
	print '-----\n',ut
	
	nfn = len(tesLbls[ii])
	# print nfn

	for wd in baseFm:
		# print '*',wd
		wdIdx = baseFm.keys().index(wd)
		nhmm = 2*len(baseFm[wd])+7*2
		
		alpha = np.zeros([nhmm,nfn+1])
		alpha1 = np.zeros([nhmm,nfn+1]) # alpha_star
		alpha2 = np.zeros([nhmm,nfn+1]) # alpha_prime
		alpha[0,0] = 1
		alpha2[0,0] = 1
		# alpha[:,0] = np.ones(nhmm)/nhmm
		# alpha2[:,0] = np.ones(nhmm)/nhmm
		bq = [1] # normalize factor (Q)
		alpha1[:,0] = alpha[:,0]/bq[0]

		bq = [0]
		alpha1 = np.log(alpha1)
		alpha2 = np.log(alpha2)
		tmpListTr = np.log(wordHMM[wd][0])
		tmpListEm = []
		for each in wordHMM[wd][1]:
			tmpListEm.append(np.log(each))

		for i in range(1,nfn+1): # compute alpha
			opIdx = lblName.index(tesLbls[ii][i-1])
			alpha2[1,i] = alpha1[0,i-1]+tmpListTr[0]+tmpListEm[0][opIdx]
			alpha2[1,i] = np.logaddexp(alpha2[1,i],alpha1[1,i-1]+tmpListTr[1]+tmpListEm[1][opIdx])
			alpha2[2,i] = alpha1[1,i-1]+tmpListTr[2]+tmpListEm[2][opIdx]
			alpha2[2,i] = np.logaddexp(alpha2[2,i],alpha1[2,i-1]+tmpListTr[3]+tmpListEm[3][opIdx])
			alpha2[3,i] = alpha1[0,i-1]+tmpListTr[5]+tmpListEm[5][opIdx]
			alpha2[4,i] = alpha1[3,i-1]+tmpListTr[6]+tmpListEm[6][opIdx]
			alpha2[5,i] = alpha1[4,i-1]+tmpListTr[7]+tmpListEm[7][opIdx]
			alpha2[6,i] = alpha1[2,i-1]+tmpListTr[4]+tmpListEm[4][opIdx]
			alpha2[6,i] = np.logaddexp(alpha2[6,i],alpha1[5,i-1]+tmpListTr[8]+tmpListEm[8][opIdx])
			alpha2[6,i] = np.logaddexp(alpha2[6,i],alpha2[3,i]+tmpListTr[9])
			alpha2[6,i] = np.logaddexp(alpha2[6,i],alpha2[4,i]+tmpListTr[10])
			alpha2[6,i] = np.logaddexp(alpha2[6,i],alpha2[5,i]+tmpListTr[11])
			for j in range(7,nhmm-7):
				if j%2==1:
					alpha2[j,i] = alpha2[j-1,i]
					alpha2[j,i] = np.logaddexp(alpha2[j,i],alpha1[j,i-1]+tmpListTr[(j-7)/2*3+13]+tmpListEm[(j-7)/2*3+13][opIdx])
				else:
					alpha2[j,i] = alpha1[j-1,i-1]+tmpListTr[(j-8)/2*3+12]+tmpListEm[(j-8)/2*3+12][opIdx]
					alpha2[j,i] = np.logaddexp(alpha2[j,i],alpha2[j-1,i]+tmpListTr[(j-8)/2*3+14])
			alpha2[-7,i] = alpha2[-8,i]
			alpha2[-6,i] = alpha1[-7,i-1]+tmpListTr[-12]+tmpListEm[-12][opIdx]
			alpha2[-6,i] = np.logaddexp(alpha2[-6,i],alpha1[-6,i-1]+tmpListTr[-11]+tmpListEm[-11][opIdx])
			alpha2[-5,i] = alpha1[-6,i-1]+tmpListTr[-10]+tmpListEm[-10][opIdx]
			alpha2[-5,i] = np.logaddexp(alpha2[-5,i],alpha1[-5,i-1]+tmpListTr[-9]+tmpListEm[-9][opIdx])
			alpha2[-4,i] = alpha1[-7,i-1]+tmpListTr[-7]+tmpListEm[-7][opIdx]
			alpha2[-3,i] = alpha1[-4,i-1]+tmpListTr[-6]+tmpListEm[-6][opIdx]
			alpha2[-2,i] = alpha1[-3,i-1]+tmpListTr[-5]+tmpListEm[-5][opIdx]
			alpha2[-1,i] = alpha1[-5,i-1]+tmpListTr[-8]+tmpListEm[-8][opIdx]
			alpha2[-1,i] = np.logaddexp(alpha2[-1,i],alpha1[-2,i-1]+tmpListTr[-4]+tmpListEm[-4][opIdx])
			alpha2[-1,i] = np.logaddexp(alpha2[-1,i],alpha2[-4,i]+tmpListTr[-3])
			alpha2[-1,i] = np.logaddexp(alpha2[-1,i],alpha2[-3,i]+tmpListTr[-2])
			alpha2[-1,i] = np.logaddexp(alpha2[-1,i],alpha2[-2,i]+tmpListTr[-1])

			sumtmp = -float('inf')
			for k in range(nhmm):
				# print '$$',alpha2[k,i]
				# print '*',sumtmp
				sumtmp = np.logaddexp(sumtmp,alpha2[k,i])
			if sumtmp!=sumtmp:
				sumtmp = -float('inf')
			bq.append(sumtmp)
			alpha1[:,i] = alpha2[:,i]-bq[i]

		# for i in range(nfn+1):
		# 	print '***** alpha_\'',i,':'
		# 	print alpha2[:,i]
		# 	print '***** Q:',bq[i]
		# 	print '***** alpha_*',i,':'
		# 	print alpha1[:,i]
		# print 'Q:',len(bq),'\n',bq

		bqpd = 0 ### log-prob of data
		for q in bq:
			bqpd = bqpd+q
		# dataProb[ii,wdIdx] = bqpd/nfn
		# if bqpd!=bqpd:
		# 	bqpd = -float('inf')
		dataProb[ii,wdIdx] = bqpd
		# print bqpd

		# break

	# break

	tmp = dataProb[ii,:]
	tmp = tmp-np.amax(tmp)
	sumtmp = -float('inf')
	for j in range(len(baseFm.keys())):
		sumtmp = np.logaddexp(sumtmp,tmp[j])
	tmp = tmp-sumtmp
	# tmp = np.exp(tmp)
	# tmp = tmp/np.sum(tmp)
	cf = np.amax(tmp)
	conf.append(math.exp(cf))
	idx = np.argmax(tmp)
	idy.append(baseFm.keys()[idx])
	print idy[ii],conf[ii]

	# break

# print dataProb
# print idy
# print conf

file = open('result_cntra.txt','w')
file.write('recongized_word confidence\n')
for ii in range(len(tesLbls)):
	line = idy[ii]+' '+str(conf[ii])+'\n'
	file.write(line)
file.close()
















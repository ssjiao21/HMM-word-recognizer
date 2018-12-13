#!/usr/bin/env python

import numpy as np
import math
import pickle
import random

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

def init():
	##### Create elementary models (1) #####
	eleHMM = {}
	for lbl in lblName:
		tmp = []
		tmp.append(np.array([0.8,0.1,0.1]))	
		tmp2 = np.ones(256)*0.5/255
		tmp2[lblName.index(lbl)] = 0.5
		tmp.append([tmp2]*2+[np.zeros(256)])
		eleHMM[lbl] = tmp[:]
	# print 'AA:\n',eleHMM['AA']

	##### Create the silence model (2) #####
	silHMM = []
	silHMM.append(np.ones(12)/2)
	tmp2 = np.ones(256)/256
	silHMM.append([tmp2]*9+[np.zeros(256)]*3)
	# print 'Silence:\n',silHMM

	return eleHMM,silHMM
##### -------------------------------------------------------------------- #####
print '---------------- START -----------------'

file = open('clsp.lblnames','r')
lblName = []
for line in file:
	lblName.append(line.rstrip('\n'))
lblName.pop(0)
file.close()
# print lblName

file = open('clsp.trnscr','r')
words = {}
voc = []
i = 0
file.readline()
for line in file:
	i = i+1
	vv = line.rstrip('\n')
	words[str(i)] = vv
	if vv not in voc:
		voc.append(vv)
file.close()
# print words
# print voc

file = open('clsp.trnlbls','r')
trnLbls = {}
i = 0
file.readline()
for line in file:
	i = i+1
	tmp = line.rstrip('\n').split(' ')
	if '' in tmp:
		tmp.remove('')
	trnLbls[str(i)] = tmp
	# print len(tmp)
file.close()
# print trnLbls

file = open('clsp.endpts','r')
positn = {}
i = 0
file.readline()
for line in file:
	i = i+1
	tmp = line.rstrip('\n').split(' ')
	tmp2 = [int(x) for x in tmp]
	positn[str(i)] = tmp2
file.close()
# print positn

heot = []
kept = []

for each in voc:
	tmpList = []
	for ii in range(len(words.keys())):
		if each==words[str(ii+1)]:
			tmpList.append(str(ii+1))
	ntot = len(tmpList)
	nheot = ntot/5
	# print len(tmpList),nheot
	heltIdx = random.sample(range(0,ntot),nheot)
	heotTmp = [tmpList[i] for i in heltIdx]
	keptTmp = [x for x in tmpList if x not in heotTmp]
	# heotTmp = tmpList[0:nheot]
	# keptTmp = tmpList[nheot:len(tmpList)]
	for idx in heotTmp:
		heot.append(idx)
	for idx in keptTmp:
		kept.append(idx)
	# print each,'\n',tmpList
	# break
# print len(heot)
# print len(kept)

##### Initialization #####
[eleHMM,silHMM] = init()

##### Pick fenonic baseforms (3) #####
baseFm = {}
for idx in kept:
	if words[idx] not in baseFm:
		tmp = trnLbls[idx][positn[idx][0]:positn[idx][1]-1]
		baseFm[words[idx]] = tmp
# print baseFm
# print 'money:',baseFm['money']
# print 'enjoy:',baseFm['enjoy']
np.save('baseform_cntra.npy',baseFm)

##### Form word HMMs (4) #####
wordHMM = HMM_word(eleHMM,silHMM)

############################## Tuning ##############################
doFlag = 1
iterN = 0
acc =[]
while doFlag:
	print '***** tuning iteration',iterN+1,'*****'

	contrEt = np.zeros([256,3])
	contrSt = np.zeros(12)
	contrEe = np.zeros([256,3,256])
	contrSe = np.zeros([12,256])

	ut = 0
	for wd in baseFm:
		print '-----\n',wd
		# narcs = 3*len(baseFm1[wd])+12*2
		nhmm = 2*len(baseFm[wd])+7*2

		for idx in kept:
			if words[idx]==wd:
				print '*',ut+1
				nfn = len(trnLbls[idx])
				# print nhmm
				# print nfn
				alpha = np.zeros([nhmm,nfn+1])
				alpha1 = np.zeros([nhmm,nfn+1]) # alpha_star
				alpha2 = np.zeros([nhmm,nfn+1]) # alpha_prime
				alpha[0,0] = 1
				alpha2[0,0] = 1
				# alpha[:,0] = np.ones(nhmm)/nhmm
				# alpha2[:,0] = np.ones(nhmm)/nhmm
				bq = [1] # normalize factor (Q)
				alpha1[:,0] = alpha[:,0]/bq[0]

				for i in range(1,nfn+1): # compute alpha
					opIdx = lblName.index(trnLbls[idx][i-1])
					alpha2[1,i] = alpha1[0,i-1]*wordHMM[wd][0][0]*wordHMM[wd][1][0][opIdx]
					alpha2[1,i] += alpha1[1,i-1]*wordHMM[wd][0][1]*wordHMM[wd][1][1][opIdx]
					alpha2[2,i] = alpha1[1,i-1]*wordHMM[wd][0][2]*wordHMM[wd][1][2][opIdx]
					alpha2[2,i] += alpha1[2,i-1]*wordHMM[wd][0][3]*wordHMM[wd][1][3][opIdx]
					alpha2[3,i] = alpha1[0,i-1]*wordHMM[wd][0][5]*wordHMM[wd][1][5][opIdx]
					alpha2[4,i] = alpha1[3,i-1]*wordHMM[wd][0][6]*wordHMM[wd][1][6][opIdx]
					alpha2[5,i] = alpha1[4,i-1]*wordHMM[wd][0][7]*wordHMM[wd][1][7][opIdx]
					alpha2[6,i] = alpha1[2,i-1]*wordHMM[wd][0][4]*wordHMM[wd][1][4][opIdx]
					alpha2[6,i] += alpha1[5,i-1]*wordHMM[wd][0][8]*wordHMM[wd][1][8][opIdx]
					alpha2[6,i] += alpha2[3,i]*wordHMM[wd][0][9]
					alpha2[6,i] += alpha2[4,i]*wordHMM[wd][0][10]
					alpha2[6,i] += alpha2[5,i]*wordHMM[wd][0][11]
					for j in range(7,nhmm-7):
						if j%2==1:
							alpha2[j,i] = alpha2[j-1,i]
							alpha2[j,i] += alpha1[j,i-1]*wordHMM[wd][0][(j-7)/2*3+13]*wordHMM[wd][1][(j-7)/2*3+13][opIdx]
						else:
							alpha2[j,i] = alpha1[j-1,i-1]*wordHMM[wd][0][(j-8)/2*3+12]*wordHMM[wd][1][(j-8)/2*3+12][opIdx]
							alpha2[j,i] += alpha2[j-1,i]*wordHMM[wd][0][(j-8)/2*3+14]
					alpha2[-7,i] = alpha2[-8,i]
					alpha2[-6,i] = alpha1[-7,i-1]*wordHMM[wd][0][-12]*wordHMM[wd][1][-12][opIdx]
					alpha2[-6,i] += alpha1[-6,i-1]*wordHMM[wd][0][-11]*wordHMM[wd][1][-11][opIdx]
					alpha2[-5,i] = alpha1[-6,i-1]*wordHMM[wd][0][-10]*wordHMM[wd][1][-10][opIdx]
					alpha2[-5,i] += alpha1[-5,i-1]*wordHMM[wd][0][-9]*wordHMM[wd][1][-9][opIdx]
					alpha2[-4,i] = alpha1[-7,i-1]*wordHMM[wd][0][-7]*wordHMM[wd][1][-7][opIdx]
					alpha2[-3,i] = alpha1[-4,i-1]*wordHMM[wd][0][-6]*wordHMM[wd][1][-6][opIdx]
					alpha2[-2,i] = alpha1[-3,i-1]*wordHMM[wd][0][-5]*wordHMM[wd][1][-5][opIdx]
					alpha2[-1,i] = alpha1[-5,i-1]*wordHMM[wd][0][-8]*wordHMM[wd][1][-8][opIdx]
					alpha2[-1,i] += alpha1[-2,i-1]*wordHMM[wd][0][-4]*wordHMM[wd][1][-4][opIdx]
					alpha2[-1,i] += alpha2[-4,i]*wordHMM[wd][0][-3]
					alpha2[-1,i] += alpha2[-3,i]*wordHMM[wd][0][-2]
					alpha2[-1,i] += alpha2[-2,i]*wordHMM[wd][0][-1]

					bq.append(np.sum(alpha2[:,i],axis=0))
					alpha1[:,i] = alpha2[:,i]/bq[i]

				# if iterN>0:
				# 	for i in range(nfn+1):
				# 		print bq[i],'============================'
				# 		for j in range(nhmm):
				# 			print j,',',i,',',alpha2[j,i]
				# 			print '*****'
				# if iterN>0:
				# 	print 'Q:',len(bq),'\n',bq

				beta = np.zeros([nhmm,nfn+1])
				beta1 = np.zeros([nhmm,nfn+1]) # beta_star
				beta2 = np.zeros([nhmm,nfn+1]) # beta_prime
				beta[:,nfn] = 1
				beta2[:,nfn] = 1
				beta1[:,nfn] = beta[:,nfn]/bq[nfn]

				for i in range(nfn,0,-1): # compute beta
					opIdx = lblName.index(trnLbls[idx][i-1])
					beta2[-2,i-1] = beta1[-1,i]*wordHMM[wd][0][-4]*wordHMM[wd][1][-4][opIdx]
					beta2[-2,i-1] += beta2[-1,i-1]*wordHMM[wd][0][-1]
					beta2[-3,i-1] = beta1[-2,i]*wordHMM[wd][0][-5]*wordHMM[wd][1][-5][opIdx]
					beta2[-3,i-1] += beta2[-1,i-1]*wordHMM[wd][0][-2]
					beta2[-4,i-1] = beta1[-3,i]*wordHMM[wd][0][-6]*wordHMM[wd][1][-6][opIdx]
					beta2[-4,i-1] += beta2[-1,i-1]*wordHMM[wd][0][-3]
					beta2[-5,i-1] = beta1[-1,i]*wordHMM[wd][0][-8]*wordHMM[wd][1][-8][opIdx]
					beta2[-5,i-1] += beta1[-5,i]*wordHMM[wd][0][-9]*wordHMM[wd][1][-9][opIdx]
					beta2[-6,i-1] = beta1[-5,i]*wordHMM[wd][0][-10]*wordHMM[wd][1][-10][opIdx]
					beta2[-6,i-1] += beta1[-6,i]*wordHMM[wd][0][-11]*wordHMM[wd][1][-11][opIdx]
					beta2[-7,i-1] = beta1[-6,i]*wordHMM[wd][0][-12]*wordHMM[wd][1][-12][opIdx]
					beta2[-7,i-1] += beta1[-4,i]*wordHMM[wd][0][-7]*wordHMM[wd][1][-7][opIdx]
					for j in range(7,nhmm-7):
						if j%2==1:
							beta2[-j-1,i-1] = beta2[-j,i-1]
						else:
							beta2[-j-1,i-1] = beta1[-j,i]*wordHMM[wd][0][-((j-8)/2*3+15)]*wordHMM[wd][1][-((j-8)/2*3+15)][opIdx]
							beta2[-j-1,i-1] += beta1[-j-1,i]*wordHMM[wd][0][-((j-8)/2*3+14)]*wordHMM[wd][1][-((j-8)/2*3+14)][opIdx]
							beta2[-j-1,i-1] += beta2[-j,i-1]*wordHMM[wd][0][-((j-8)/2*3+13)]
					beta2[6,i-1] = beta2[7,i-1]
					beta2[5,i-1] = beta1[6,i]*wordHMM[wd][0][8]*wordHMM[wd][1][8][opIdx]
					beta2[5,i-1] += beta2[6,i-1]*wordHMM[wd][0][11]
					beta2[4,i-1] = beta1[5,i]*wordHMM[wd][0][7]*wordHMM[wd][1][7][opIdx]
					beta2[4,i-1] += beta2[6,i-1]*wordHMM[wd][0][10]
					beta2[3,i-1] = beta1[4,i]*wordHMM[wd][0][6]*wordHMM[wd][1][6][opIdx]
					beta2[3,i-1] += beta2[6,i-1]*wordHMM[wd][0][9]
					beta2[2,i-1] = beta1[6,i]*wordHMM[wd][0][4]*wordHMM[wd][1][4][opIdx]
					beta2[2,i-1] += beta1[2,i]*wordHMM[wd][0][3]*wordHMM[wd][1][3][opIdx]
					beta2[1,i-1] = beta1[2,i]*wordHMM[wd][0][2]*wordHMM[wd][1][2][opIdx]
					beta2[1,i-1] += beta1[1,i]*wordHMM[wd][0][1]*wordHMM[wd][1][1][opIdx]
					beta2[0,i-1] = beta1[1,i]*wordHMM[wd][0][0]*wordHMM[wd][1][0][opIdx]
					beta2[0,i-1] += beta1[3,i]*wordHMM[wd][0][5]*wordHMM[wd][1][5][opIdx]
					
					beta1[:,i-1] = beta2[:,i-1]/bq[i-1]

				# for i in range(nfn+1):
					# print alpha2[:,i]
					# print '***** alpha_*',i,':'
					# print alpha1[:,i]
					# print '***** beta_\'',i,':'	
					# print beta2[:,i]
					# print '***** beta_*',i,':'
					# print beta1[:,i]
			
				# print 'Q:',len(bq),'\n',bq
				probtmpS = np.zeros([12,256,nfn])
				probtmpE = np.zeros([256,3,256,nfn])
				for i in range(nfn): # compute posterior prob 
					opIdx = lblName.index(trnLbls[idx][i])
					probtmpS[0,opIdx,i] += alpha1[0,i]*wordHMM[wd][0][0]*wordHMM[wd][1][0][opIdx]*beta1[1,i+1]
					probtmpS[1,opIdx,i] += alpha1[1,i]*wordHMM[wd][0][1]*wordHMM[wd][1][1][opIdx]*beta1[1,i+1]
					probtmpS[2,opIdx,i] += alpha1[1,i]*wordHMM[wd][0][2]*wordHMM[wd][1][2][opIdx]*beta1[2,i+1]
					probtmpS[3,opIdx,i] += alpha1[2,i]*wordHMM[wd][0][3]*wordHMM[wd][1][3][opIdx]*beta1[2,i+1]
					probtmpS[4,opIdx,i] += alpha1[2,i]*wordHMM[wd][0][4]*wordHMM[wd][1][4][opIdx]*beta1[6,i+1]
					probtmpS[5,opIdx,i] += alpha1[0,i]*wordHMM[wd][0][5]*wordHMM[wd][1][5][opIdx]*beta1[3,i+1]
					probtmpS[6,opIdx,i] += alpha1[3,i]*wordHMM[wd][0][6]*wordHMM[wd][1][6][opIdx]*beta1[4,i+1]
					probtmpS[7,opIdx,i] += alpha1[4,i]*wordHMM[wd][0][7]*wordHMM[wd][1][7][opIdx]*beta1[5,i+1]
					probtmpS[8,opIdx,i] += alpha1[5,i]*wordHMM[wd][0][8]*wordHMM[wd][1][8][opIdx]*beta1[6,i+1]
					probtmpS[9,opIdx,i] += alpha1[3,i]*wordHMM[wd][0][9]*beta1[6,i]*bq[i]
					probtmpS[10,opIdx,i] += alpha1[4,i]*wordHMM[wd][0][10]*beta1[6,i]*bq[i]
					probtmpS[11,opIdx,i] += alpha1[5,i]*wordHMM[wd][0][11]*beta1[6,i]*bq[i]
					for j in range(7,nhmm-7):
						if j%2==1:
							mdIdx = lblName.index(baseFm[wd][(j-7)/2])
							probtmpE[mdIdx,0,opIdx,i] += alpha1[j,i]*wordHMM[wd][0][(j-7)/2*3+12]*wordHMM[wd][1][(j-7)/2*3+12][opIdx]*beta1[j+1,i+1]
							probtmpE[mdIdx,1,opIdx,i] += alpha1[j,i]*wordHMM[wd][0][(j-7)/2*3+13]*wordHMM[wd][1][(j-7)/2*3+13][opIdx]*beta1[j,i+1]
							probtmpE[mdIdx,2,opIdx,i] += alpha1[j,i]*wordHMM[wd][0][(j-7)/2*3+14]*beta1[j+1,i]*bq[i]
					probtmpS[0,opIdx,i] += alpha1[-7,i]*wordHMM[wd][0][0]*wordHMM[wd][1][0][opIdx]*beta1[-6,i+1]
					probtmpS[1,opIdx,i] += alpha1[-6,i]*wordHMM[wd][0][1]*wordHMM[wd][1][1][opIdx]*beta1[-6,i+1]
					probtmpS[2,opIdx,i] += alpha1[-6,i]*wordHMM[wd][0][2]*wordHMM[wd][1][2][opIdx]*beta1[-5,i+1]
					probtmpS[3,opIdx,i] += alpha1[-5,i]*wordHMM[wd][0][3]*wordHMM[wd][1][3][opIdx]*beta1[-5,i+1]
					probtmpS[4,opIdx,i] += alpha1[-5,i]*wordHMM[wd][0][4]*wordHMM[wd][1][4][opIdx]*beta1[-1,i+1]
					probtmpS[5,opIdx,i] += alpha1[-7,i]*wordHMM[wd][0][5]*wordHMM[wd][1][5][opIdx]*beta1[-4,i+1]
					probtmpS[6,opIdx,i] += alpha1[-4,i]*wordHMM[wd][0][6]*wordHMM[wd][1][6][opIdx]*beta1[-3,i+1]
					probtmpS[7,opIdx,i] += alpha1[-3,i]*wordHMM[wd][0][7]*wordHMM[wd][1][7][opIdx]*beta1[-2,i+1]
					probtmpS[8,opIdx,i] += alpha1[-2,i]*wordHMM[wd][0][8]*wordHMM[wd][1][8][opIdx]*beta1[-1,i+1]
					probtmpS[9,opIdx,i] += alpha1[-4,i]*wordHMM[wd][0][9]*beta1[-1,i]*bq[i]
					probtmpS[10,opIdx,i] += alpha1[-3,i]*wordHMM[wd][0][10]*beta1[-1,i]*bq[i]
					probtmpS[11,opIdx,i] += alpha1[-2,i]*wordHMM[wd][0][11]*beta1[-1,i]*bq[i]	
				# print np.sum(probtmpS)
				# print np.sum(probtmpE)

				contrEt += np.sum(probtmpE,axis=(2,3))
				contrSt += np.sum(probtmpS,axis=(1,2))
				contrEe += np.sum(probtmpE,axis=3)
				contrSe += np.sum(probtmpS,axis=2)

				# bqpd = 0 ### log-prob of data
				# dProbtmp = dataProb[ut][:]
				# for q in bq:
				# 	bqpd = bqpd+math.log(q)
				# dProbtmp.append(bqpd)
				# dataProb[ut] = dProbtmp[:]
				ut = ut+1
				
				# break
		# break Forward-back pass
	# break

	trsi1 = contrEt[:]	#ele
	trsi2 = contrSt[:]	#sil
	emis1 = contrEe[:]	#ele
	emis2 = contrSe[:]	#sil

	# print '***** eleHMM transition\n',trsi1
	# print '***** eleHMM emission'
	# for i in range(256):
	# 	print '$$',i,'\n',np.transpose(emis1[i,:,:])
	# 	# print '$$',i,'\n',emis1[:,:,i]
	# print '***** silHMM transition\n',trsi2
	# print '***** silHMM emission'
	# for i in range(12):
	# 	print '$$',i,'\n',emis2[i,:]
	# # for i in range(256):
	# # 	print '$$',i,'\n',emis2[:,i]

	for i in range(256):
		sumtemp = np.sum(trsi1[i,:])
		if sumtemp!=0:
			trsi1[i,:] = trsi1[i,:]/sumtemp
			eleHMM[lblName[i]][0] = trsi1[i,:]
		for j in range(2):
			sumtemp = np.sum(emis1[i,j,:])
			if sumtemp!=0:
				emis1[i,j,:] = emis1[i,j,:]/sumtemp
				eleHMM[lblName[i]][1][j] = emis1[i,j,:]

	sumtmp = trsi2[0]+trsi2[5]
	if sumtemp!=0:
		trsi2[0] = trsi2[0]/sumtmp
		trsi2[5] = trsi2[5]/sumtmp
		silHMM[0][0] = trsi2[0]
		silHMM[0][5] = trsi2[5]
	sumtmp = trsi2[1]+trsi2[2]
	if sumtemp!=0:
		trsi2[1] = trsi2[1]/sumtmp
		trsi2[2] = trsi2[2]/sumtmp
		silHMM[0][1] = trsi2[1]
		silHMM[0][2] = trsi2[2]
	sumtmp = trsi2[3]+trsi2[4]
	if sumtemp!=0:
		trsi2[3] = trsi2[3]/sumtmp
		trsi2[4] = trsi2[4]/sumtmp
		silHMM[0][3] = trsi2[3]
		silHMM[0][4] = trsi2[4]
	sumtmp = trsi2[6]+trsi2[9]
	if sumtemp!=0:
		trsi2[6] = trsi2[6]/sumtmp
		trsi2[9] = trsi2[9]/sumtmp
		silHMM[0][6] = trsi2[6]
		silHMM[0][9] = trsi2[9]
	sumtmp = trsi2[7]+trsi2[10]
	if sumtemp!=0:
		trsi2[7] = trsi2[7]/sumtmp
		trsi2[10] = trsi2[10]/sumtmp
		silHMM[0][7] = trsi2[7]
		silHMM[0][10] = trsi2[10]
	sumtmp = trsi2[8]+trsi2[11]
	if sumtemp!=0:
		trsi2[8] = trsi2[8]/sumtmp
		trsi2[11] = trsi2[11]/sumtmp
		silHMM[0][8] = trsi2[8]
		silHMM[0][11] = trsi2[11]
	for k in range(9):
		sumtemp = np.sum(emis2[k,:])
		if sumtemp!=0:
			emis2[k,:] = emis2[k,:]/sumtemp
			silHMM[1][k] = emis2[k,:]

	wordHMM = HMM_word(eleHMM,silHMM)
	# print 'enjoy:',wordHMM['enjoy']

	iterN = iterN+1

	##### Compute tunning accuracy #####

	dataProb = np.zeros([len(heot),len(baseFm.keys())])
	# print dataProb.shape
	idy = []
	conf = []

	ut = 0
	for idx in heot:
		ii = heot.index(idx)
		ut = ut+1
		print '-----\n',ut
		
		nfn = len(trnLbls[idx])
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
				opIdx = lblName.index(trnLbls[idx][i-1])
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
			# dataProb[idx,wdIdx] = bqpd/nfn
			# if bqpd!=bqpd:
			# 	bqpd = -float('inf')
			dataProb[ii,wdIdx] = bqpd
			# print bqpd

			# break

		# if ii==2:
		# 	break

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

	corr = 0
	for idx in heot:
		if words[idx] == idy[heot.index(idx)]:
			corr = corr+1
	# print corr
	corr = corr/float(len(heot))
	print '## Accuracy:',corr
	acc.append(corr)

	if iterN>1:
		if acc[len(acc)-1] <= acc[len(acc)-2]:
			doFlag = 0
		for jj in range(len(acc)):
			if acc[jj]==max(acc):
				noi = jj+1
				break

	# if iterN==1:
	# 	break

print '----------------------------------------'
print 'Tuning accuracy:',acc
print 'Number of iterations required:',noi
print '----------------------------------------'

##### Initialization #####
[eleHMM,silHMM] = init()
wordHMM = HMM_word(eleHMM,silHMM)

############################## Training ##############################
dataProb = []
for k in words:
	dataProb.append([])
dataProbT = []

iterN = 0
while 1:
	print '***** training iteration',iterN+1,'*****'

	contrEt = np.zeros([256,3])
	contrSt = np.zeros(12)
	contrEe = np.zeros([256,3,256])
	contrSe = np.zeros([12,256])

	nfntot = 0
	ut = 0
	for wd in baseFm:
		print '-----\n',wd
		# narcs = 3*len(baseFm1[wd])+12*2
		nhmm = 2*len(baseFm[wd])+7*2

		for ii in range(len(words.keys())):
			idx = str(ii+1)
			if words[idx]==wd:
				print '*',ut+1
				nfn = len(trnLbls[idx])
				nfntot += nfn
				# print nhmm
				# print nfn
				alpha = np.zeros([nhmm,nfn+1])
				alpha1 = np.zeros([nhmm,nfn+1]) # alpha_star
				alpha2 = np.zeros([nhmm,nfn+1]) # alpha_prime
				alpha[0,0] = 1
				alpha2[0,0] = 1
				# alpha[:,0] = np.ones(nhmm)/nhmm
				# alpha2[:,0] = np.ones(nhmm)/nhmm
				bq = [1] # normalize factor (Q)
				alpha1[:,0] = alpha[:,0]/bq[0]

				for i in range(1,nfn+1): # compute alpha
					opIdx = lblName.index(trnLbls[idx][i-1])
					alpha2[1,i] = alpha1[0,i-1]*wordHMM[wd][0][0]*wordHMM[wd][1][0][opIdx]
					alpha2[1,i] += alpha1[1,i-1]*wordHMM[wd][0][1]*wordHMM[wd][1][1][opIdx]
					alpha2[2,i] = alpha1[1,i-1]*wordHMM[wd][0][2]*wordHMM[wd][1][2][opIdx]
					alpha2[2,i] += alpha1[2,i-1]*wordHMM[wd][0][3]*wordHMM[wd][1][3][opIdx]
					alpha2[3,i] = alpha1[0,i-1]*wordHMM[wd][0][5]*wordHMM[wd][1][5][opIdx]
					alpha2[4,i] = alpha1[3,i-1]*wordHMM[wd][0][6]*wordHMM[wd][1][6][opIdx]
					alpha2[5,i] = alpha1[4,i-1]*wordHMM[wd][0][7]*wordHMM[wd][1][7][opIdx]
					alpha2[6,i] = alpha1[2,i-1]*wordHMM[wd][0][4]*wordHMM[wd][1][4][opIdx]
					alpha2[6,i] += alpha1[5,i-1]*wordHMM[wd][0][8]*wordHMM[wd][1][8][opIdx]
					alpha2[6,i] += alpha2[3,i]*wordHMM[wd][0][9]
					alpha2[6,i] += alpha2[4,i]*wordHMM[wd][0][10]
					alpha2[6,i] += alpha2[5,i]*wordHMM[wd][0][11]
					for j in range(7,nhmm-7):
						if j%2==1:
							alpha2[j,i] = alpha2[j-1,i]
							alpha2[j,i] += alpha1[j,i-1]*wordHMM[wd][0][(j-7)/2*3+13]*wordHMM[wd][1][(j-7)/2*3+13][opIdx]
						else:
							alpha2[j,i] = alpha1[j-1,i-1]*wordHMM[wd][0][(j-8)/2*3+12]*wordHMM[wd][1][(j-8)/2*3+12][opIdx]
							alpha2[j,i] += alpha2[j-1,i]*wordHMM[wd][0][(j-8)/2*3+14]
					alpha2[-7,i] = alpha2[-8,i]
					alpha2[-6,i] = alpha1[-7,i-1]*wordHMM[wd][0][-12]*wordHMM[wd][1][-12][opIdx]
					alpha2[-6,i] += alpha1[-6,i-1]*wordHMM[wd][0][-11]*wordHMM[wd][1][-11][opIdx]
					alpha2[-5,i] = alpha1[-6,i-1]*wordHMM[wd][0][-10]*wordHMM[wd][1][-10][opIdx]
					alpha2[-5,i] += alpha1[-5,i-1]*wordHMM[wd][0][-9]*wordHMM[wd][1][-9][opIdx]
					alpha2[-4,i] = alpha1[-7,i-1]*wordHMM[wd][0][-7]*wordHMM[wd][1][-7][opIdx]
					alpha2[-3,i] = alpha1[-4,i-1]*wordHMM[wd][0][-6]*wordHMM[wd][1][-6][opIdx]
					alpha2[-2,i] = alpha1[-3,i-1]*wordHMM[wd][0][-5]*wordHMM[wd][1][-5][opIdx]
					alpha2[-1,i] = alpha1[-5,i-1]*wordHMM[wd][0][-8]*wordHMM[wd][1][-8][opIdx]
					alpha2[-1,i] += alpha1[-2,i-1]*wordHMM[wd][0][-4]*wordHMM[wd][1][-4][opIdx]
					alpha2[-1,i] += alpha2[-4,i]*wordHMM[wd][0][-3]
					alpha2[-1,i] += alpha2[-3,i]*wordHMM[wd][0][-2]
					alpha2[-1,i] += alpha2[-2,i]*wordHMM[wd][0][-1]

					bq.append(np.sum(alpha2[:,i],axis=0))
					alpha1[:,i] = alpha2[:,i]/bq[i]

				# if iterN>0:
				# 	for i in range(nfn+1):
				# 		print bq[i],'============================'
				# 		for j in range(nhmm):
				# 			print j,',',i,',',alpha2[j,i]
				# 			print '*****'
				# if iterN>0:
				# 	print 'Q:',len(bq),'\n',bq

				beta = np.zeros([nhmm,nfn+1])
				beta1 = np.zeros([nhmm,nfn+1]) # beta_star
				beta2 = np.zeros([nhmm,nfn+1]) # beta_prime
				beta[:,nfn] = 1
				beta2[:,nfn] = 1
				beta1[:,nfn] = beta[:,nfn]/bq[nfn]

				for i in range(nfn,0,-1): # compute beta
					opIdx = lblName.index(trnLbls[idx][i-1])
					beta2[-2,i-1] = beta1[-1,i]*wordHMM[wd][0][-4]*wordHMM[wd][1][-4][opIdx]
					beta2[-2,i-1] += beta2[-1,i-1]*wordHMM[wd][0][-1]
					beta2[-3,i-1] = beta1[-2,i]*wordHMM[wd][0][-5]*wordHMM[wd][1][-5][opIdx]
					beta2[-3,i-1] += beta2[-1,i-1]*wordHMM[wd][0][-2]
					beta2[-4,i-1] = beta1[-3,i]*wordHMM[wd][0][-6]*wordHMM[wd][1][-6][opIdx]
					beta2[-4,i-1] += beta2[-1,i-1]*wordHMM[wd][0][-3]
					beta2[-5,i-1] = beta1[-1,i]*wordHMM[wd][0][-8]*wordHMM[wd][1][-8][opIdx]
					beta2[-5,i-1] += beta1[-5,i]*wordHMM[wd][0][-9]*wordHMM[wd][1][-9][opIdx]
					beta2[-6,i-1] = beta1[-5,i]*wordHMM[wd][0][-10]*wordHMM[wd][1][-10][opIdx]
					beta2[-6,i-1] += beta1[-6,i]*wordHMM[wd][0][-11]*wordHMM[wd][1][-11][opIdx]
					beta2[-7,i-1] = beta1[-6,i]*wordHMM[wd][0][-12]*wordHMM[wd][1][-12][opIdx]
					beta2[-7,i-1] += beta1[-4,i]*wordHMM[wd][0][-7]*wordHMM[wd][1][-7][opIdx]
					for j in range(7,nhmm-7):
						if j%2==1:
							beta2[-j-1,i-1] = beta2[-j,i-1]
						else:
							beta2[-j-1,i-1] = beta1[-j,i]*wordHMM[wd][0][-((j-8)/2*3+15)]*wordHMM[wd][1][-((j-8)/2*3+15)][opIdx]
							beta2[-j-1,i-1] += beta1[-j-1,i]*wordHMM[wd][0][-((j-8)/2*3+14)]*wordHMM[wd][1][-((j-8)/2*3+14)][opIdx]
							beta2[-j-1,i-1] += beta2[-j,i-1]*wordHMM[wd][0][-((j-8)/2*3+13)]
					beta2[6,i-1] = beta2[7,i-1]
					beta2[5,i-1] = beta1[6,i]*wordHMM[wd][0][8]*wordHMM[wd][1][8][opIdx]
					beta2[5,i-1] += beta2[6,i-1]*wordHMM[wd][0][11]
					beta2[4,i-1] = beta1[5,i]*wordHMM[wd][0][7]*wordHMM[wd][1][7][opIdx]
					beta2[4,i-1] += beta2[6,i-1]*wordHMM[wd][0][10]
					beta2[3,i-1] = beta1[4,i]*wordHMM[wd][0][6]*wordHMM[wd][1][6][opIdx]
					beta2[3,i-1] += beta2[6,i-1]*wordHMM[wd][0][9]
					beta2[2,i-1] = beta1[6,i]*wordHMM[wd][0][4]*wordHMM[wd][1][4][opIdx]
					beta2[2,i-1] += beta1[2,i]*wordHMM[wd][0][3]*wordHMM[wd][1][3][opIdx]
					beta2[1,i-1] = beta1[2,i]*wordHMM[wd][0][2]*wordHMM[wd][1][2][opIdx]
					beta2[1,i-1] += beta1[1,i]*wordHMM[wd][0][1]*wordHMM[wd][1][1][opIdx]
					beta2[0,i-1] = beta1[1,i]*wordHMM[wd][0][0]*wordHMM[wd][1][0][opIdx]
					beta2[0,i-1] += beta1[3,i]*wordHMM[wd][0][5]*wordHMM[wd][1][5][opIdx]
					
					beta1[:,i-1] = beta2[:,i-1]/bq[i-1]

				# for i in range(nfn+1):
					# print alpha2[:,i]
					# print '***** alpha_*',i,':'
					# print alpha1[:,i]
					# print '***** beta_\'',i,':'	
					# print beta2[:,i]
					# print '***** beta_*',i,':'
					# print beta1[:,i]
			
				# print 'Q:',len(bq),'\n',bq
				probtmpS = np.zeros([12,256,nfn])
				probtmpE = np.zeros([256,3,256,nfn])
				for i in range(nfn): # compute posterior prob 
					opIdx = lblName.index(trnLbls[idx][i])
					probtmpS[0,opIdx,i] += alpha1[0,i]*wordHMM[wd][0][0]*wordHMM[wd][1][0][opIdx]*beta1[1,i+1]
					probtmpS[1,opIdx,i] += alpha1[1,i]*wordHMM[wd][0][1]*wordHMM[wd][1][1][opIdx]*beta1[1,i+1]
					probtmpS[2,opIdx,i] += alpha1[1,i]*wordHMM[wd][0][2]*wordHMM[wd][1][2][opIdx]*beta1[2,i+1]
					probtmpS[3,opIdx,i] += alpha1[2,i]*wordHMM[wd][0][3]*wordHMM[wd][1][3][opIdx]*beta1[2,i+1]
					probtmpS[4,opIdx,i] += alpha1[2,i]*wordHMM[wd][0][4]*wordHMM[wd][1][4][opIdx]*beta1[6,i+1]
					probtmpS[5,opIdx,i] += alpha1[0,i]*wordHMM[wd][0][5]*wordHMM[wd][1][5][opIdx]*beta1[3,i+1]
					probtmpS[6,opIdx,i] += alpha1[3,i]*wordHMM[wd][0][6]*wordHMM[wd][1][6][opIdx]*beta1[4,i+1]
					probtmpS[7,opIdx,i] += alpha1[4,i]*wordHMM[wd][0][7]*wordHMM[wd][1][7][opIdx]*beta1[5,i+1]
					probtmpS[8,opIdx,i] += alpha1[5,i]*wordHMM[wd][0][8]*wordHMM[wd][1][8][opIdx]*beta1[6,i+1]
					probtmpS[9,opIdx,i] += alpha1[3,i]*wordHMM[wd][0][9]*beta1[6,i]*bq[i]
					probtmpS[10,opIdx,i] += alpha1[4,i]*wordHMM[wd][0][10]*beta1[6,i]*bq[i]
					probtmpS[11,opIdx,i] += alpha1[5,i]*wordHMM[wd][0][11]*beta1[6,i]*bq[i]
					for j in range(7,nhmm-7):
						if j%2==1:
							mdIdx = lblName.index(baseFm[wd][(j-7)/2])
							probtmpE[mdIdx,0,opIdx,i] += alpha1[j,i]*wordHMM[wd][0][(j-7)/2*3+12]*wordHMM[wd][1][(j-7)/2*3+12][opIdx]*beta1[j+1,i+1]
							probtmpE[mdIdx,1,opIdx,i] += alpha1[j,i]*wordHMM[wd][0][(j-7)/2*3+13]*wordHMM[wd][1][(j-7)/2*3+13][opIdx]*beta1[j,i+1]
							probtmpE[mdIdx,2,opIdx,i] += alpha1[j,i]*wordHMM[wd][0][(j-7)/2*3+14]*beta1[j+1,i]*bq[i]
					probtmpS[0,opIdx,i] += alpha1[-7,i]*wordHMM[wd][0][0]*wordHMM[wd][1][0][opIdx]*beta1[-6,i+1]
					probtmpS[1,opIdx,i] += alpha1[-6,i]*wordHMM[wd][0][1]*wordHMM[wd][1][1][opIdx]*beta1[-6,i+1]
					probtmpS[2,opIdx,i] += alpha1[-6,i]*wordHMM[wd][0][2]*wordHMM[wd][1][2][opIdx]*beta1[-5,i+1]
					probtmpS[3,opIdx,i] += alpha1[-5,i]*wordHMM[wd][0][3]*wordHMM[wd][1][3][opIdx]*beta1[-5,i+1]
					probtmpS[4,opIdx,i] += alpha1[-5,i]*wordHMM[wd][0][4]*wordHMM[wd][1][4][opIdx]*beta1[-1,i+1]
					probtmpS[5,opIdx,i] += alpha1[-7,i]*wordHMM[wd][0][5]*wordHMM[wd][1][5][opIdx]*beta1[-4,i+1]
					probtmpS[6,opIdx,i] += alpha1[-4,i]*wordHMM[wd][0][6]*wordHMM[wd][1][6][opIdx]*beta1[-3,i+1]
					probtmpS[7,opIdx,i] += alpha1[-3,i]*wordHMM[wd][0][7]*wordHMM[wd][1][7][opIdx]*beta1[-2,i+1]
					probtmpS[8,opIdx,i] += alpha1[-2,i]*wordHMM[wd][0][8]*wordHMM[wd][1][8][opIdx]*beta1[-1,i+1]
					probtmpS[9,opIdx,i] += alpha1[-4,i]*wordHMM[wd][0][9]*beta1[-1,i]*bq[i]
					probtmpS[10,opIdx,i] += alpha1[-3,i]*wordHMM[wd][0][10]*beta1[-1,i]*bq[i]
					probtmpS[11,opIdx,i] += alpha1[-2,i]*wordHMM[wd][0][11]*beta1[-1,i]*bq[i]	
				# print np.sum(probtmpS)
				# print np.sum(probtmpE)

				contrEt += np.sum(probtmpE,axis=(2,3))
				contrSt += np.sum(probtmpS,axis=(1,2))
				contrEe += np.sum(probtmpE,axis=3)
				contrSe += np.sum(probtmpS,axis=2)

				bqpd = 0 ### log-prob of data
				dProbtmp = dataProb[ut][:]
				for q in bq:
					bqpd = bqpd+math.log(q)
				dProbtmp.append(bqpd)
				dataProb[ut] = dProbtmp[:]
				ut = ut+1
				
				# break
		# break
	# break

	trsi1 = contrEt[:]	#ele
	trsi2 = contrSt[:]	#sil
	emis1 = contrEe[:]	#ele
	emis2 = contrSe[:]	#sil

	# print '***** eleHMM transition\n',trsi1
	# print '***** eleHMM emission'
	# for i in range(256):
	# 	print '$$',i,'\n',np.transpose(emis1[i,:,:])
	# 	# print '$$',i,'\n',emis1[:,:,i]
	# print '***** silHMM transition\n',trsi2
	# print '***** silHMM emission'
	# for i in range(12):
	# 	print '$$',i,'\n',emis2[i,:]
	# # for i in range(256):
	# # 	print '$$',i,'\n',emis2[:,i]

	for i in range(256):
		sumtemp = np.sum(trsi1[i,:])
		if sumtemp!=0:
			trsi1[i,:] = trsi1[i,:]/sumtemp
			eleHMM[lblName[i]][0] = trsi1[i,:]
		for j in range(2):
			sumtemp = np.sum(emis1[i,j,:])
			if sumtemp!=0:
				emis1[i,j,:] = emis1[i,j,:]/sumtemp
				eleHMM[lblName[i]][1][j] = emis1[i,j,:]

	sumtmp = trsi2[0]+trsi2[5]
	if sumtemp!=0:
		trsi2[0] = trsi2[0]/sumtmp
		trsi2[5] = trsi2[5]/sumtmp
		silHMM[0][0] = trsi2[0]
		silHMM[0][5] = trsi2[5]
	sumtmp = trsi2[1]+trsi2[2]
	if sumtemp!=0:
		trsi2[1] = trsi2[1]/sumtmp
		trsi2[2] = trsi2[2]/sumtmp
		silHMM[0][1] = trsi2[1]
		silHMM[0][2] = trsi2[2]
	sumtmp = trsi2[3]+trsi2[4]
	if sumtemp!=0:
		trsi2[3] = trsi2[3]/sumtmp
		trsi2[4] = trsi2[4]/sumtmp
		silHMM[0][3] = trsi2[3]
		silHMM[0][4] = trsi2[4]
	sumtmp = trsi2[6]+trsi2[9]
	if sumtemp!=0:
		trsi2[6] = trsi2[6]/sumtmp
		trsi2[9] = trsi2[9]/sumtmp
		silHMM[0][6] = trsi2[6]
		silHMM[0][9] = trsi2[9]
	sumtmp = trsi2[7]+trsi2[10]
	if sumtemp!=0:
		trsi2[7] = trsi2[7]/sumtmp
		trsi2[10] = trsi2[10]/sumtmp
		silHMM[0][7] = trsi2[7]
		silHMM[0][10] = trsi2[10]
	sumtmp = trsi2[8]+trsi2[11]
	if sumtemp!=0:
		trsi2[8] = trsi2[8]/sumtmp
		trsi2[11] = trsi2[11]/sumtmp
		silHMM[0][8] = trsi2[8]
		silHMM[0][11] = trsi2[11]
	for k in range(9):
		sumtemp = np.sum(emis2[k,:])
		if sumtemp!=0:
			emis2[k,:] = emis2[k,:]/sumtemp
			silHMM[1][k] = emis2[k,:]

	wordHMM = HMM_word(eleHMM,silHMM)
	# print 'enjoy:',wordHMM['enjoy']

	iterN = iterN+1

	tmp1 = 0
	for k in range(len(dataProb)):
		tmp1 += dataProb[k][len(dataProb[k])-1]
	tmp1 = tmp1/nfntot
	print 'Average log-prob of data:',tmp1
	dataProbT.append(tmp1)

	if iterN==noi:
		break


# print '===== eleHMM ====='
# for lb in lblName:
# 	print '*',lb
# 	print 'transition prob:\n',eleHMM[lb][0],'\nemission prob:'
# 	for i in range(len(eleHMM[lb][1])):
# 		print '$ transition',i+1,'\n',eleHMM[lb][1][i]
# print '===== silHMM ====='
# print 'transition prob:\n',silHMM[0],'\nemission prob:'
# for i in range(len(silHMM[1])):
# 	print 'transition',i+1,'\n',silHMM[1][i]


np.save('eleHMM_cntra.npy',eleHMM)
with open('silHMM_cntra','wb') as f:
    pickle.dump(silHMM,f)
np.savetxt('dataProb_cntra.csv',dataProbT,delimiter=',',fmt='%s')
# np.savetxt('acc_cntra.csv',acc,delimiter=',',fmt='%s')






















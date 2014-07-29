#!/usr/bin/env python
#encoding=utf-8

import numpy as np
import random
import sys, json, os
import matplotlib.pyplot as plt
from CCFCore import CCFHandler
import pdb, time


class testRACF(object):#RA<RACF<CF
	"""a typical test framework, not general, but just for my environment
	we applied the coarse-graining method similar to what is used in
	M. Blattner, Y.-C. Zhang, and S. Maslov, Physica A 373, 753
	(2007).: A movie has been collected by a user iff the giving rating is at least 3.
	using 10-fold cross-validation"""
	def __init__(self, testRatio=0.8):
		self.__RatingThreshold = 3#rating >=3 means collected
		self.__nfold = 10#using 10-fold cross-validation
		self.__algoType_CF = 0#Collaborative Filtering algorithm
		self.__algoType_RA = 1#Resource Allocation algorithm
		self.__algoType_RACF = 2#Resource Allocation based Collaborative Filtering algorithm
		# self.__testRatio = testRatio#the ratio of items used as test set is self.__trainRatio
		self.__userList = []
		self.__itemList = []
		self.__filePath = "./dataset/ml-100k/u.data"
		self.__dataPreProcess()	


	def __dataPreProcess(self):
		"""standardize the raw datas from local disks, store as a matrix form"""
		try:#get datas from local disk
			with open(self.__filePath, 'r') as f:
				self.__instanceList = []
				tempLine = f.readline()
				while(tempLine):
					temp = tempLine.split('\t')[:3]#user id | item id | rating
					if int(temp[2]) >= self.__RatingThreshold:#re-write ratings to {0, 1}
						temp[2] = 1
					else:
						temp[2] = 0
					self.__userList.append(temp[0])
					self.__itemList.append(temp[1])
					self.__instanceList.append(temp)
					tempLine = f.readline()
		except:
			print "get local datas error !"
			sys.exit()
		f.close()

		#get user list & item list
		self.__userList = list(set(self.__userList))#remove redundance
		self.__itemList = list(set(self.__itemList))
		self.__userNum = len(self.__userList)
		self.__itemNum = len(self.__itemList)
		self.__instanceNum = len(self.__instanceList)


	def __creat_TrainingTestData(self, nfold, ifold=0):
		#########################
		#re-organize the dataset(self.__instanceList)
		#to do
		#########################
		numPerFold = self.__instanceNum/nfold
		adjacentMatrix = np.zeros([self.__itemNum, self.__userNum])
		userItemTestList = [{} for each in range(self.__userNum)]

		#create test dataset
		temp = (ifold==nfold-1) and self.__instanceNum or (ifold+1)*numPerFold
		for each in self.__instanceList[ifold*numPerFold:temp]:
			userItemTestList[self.__userList.index(each[0])][each[1]] = each[2]#store like this structure: [{item0:rating0, item1:rating1,...},{},...]
		
		#create training dataset
		for each in (self.__instanceList[0:ifold*numPerFold] + self.__instanceList[temp:]):
			if each[2] == 1:
				adjacentMatrix[self.__itemList.index(each[1])][self.__userList.index(each[0])] = 1
		return [adjacentMatrix, userItemTestList]

	def nfold_cross_validation(self, nfold):
		rst_ARS = {self.__algoType_CF:[], self.__algoType_RA:[], self.__algoType_RACF:[]}
		for ifold in range(nfold):
			trainingTestData = self.__creat_TrainingTestData(nfold, ifold)
			for algoType in range(3):
				if algoType == self.__algoType_CF:
					rst_ARS[self.__algoType_CF].append(self.__test(trainingTestData, algoType))
				elif algoType == self.__algoType_RA:
					beta = 0.8#the optimal value of beta is 0.8 for this dataset
					rst_ARS[self.__algoType_RA].append(self.__test(trainingTestData, algoType, beta))
				elif algoType == self.__algoType_RACF:
					beta = -1.9#the optimal value of beta is -1.9 for this dataset
					weight_THRESHOLD = [1,27]
					# print 'beta'
					# print beta
					# print "weight_THRESHOLD"
					# print weight_THRESHOLD
					rst_ARS[self.__algoType_RACF].append(self.__test(trainingTestData, algoType, beta, weight_THRESHOLD))
				else:
					pass
		return rst_ARS

	def __test(self, trainingTestData, algorithmType, Beta=0, weight_THRESHOLD=[]):#RA Beta=0.8 RACF -2.0
		adjacentMatrix, userItemTestList = trainingTestData
		CCF = CCFHandler(adjacentMatrix)

		#test CF
		if algorithmType == self.__algoType_CF:
			aveRScore = 0.0#average ranking score
			tempUserNum = self.__userNum
			CCF.set_PesudoAM(0)#not using RA algorithm
			for eachUser in range(self.__userNum):
				RS, flag = self.__single_test(eachUser, CCF, userItemTestList, algorithmType)
				aveRScore += RS
				tempUserNum += flag
			if tempUserNum != 0:
				aveRScore /= tempUserNum
			print "CF"
			print aveRScore


		#test RACF
		if algorithmType == self.__algoType_RACF:
			aveRScore = 0.0#average ranking score
			tempUserNum = self.__userNum
			CCF.set_PesudoAM(1, Beta)#using RA algorithm
			for eachUser in range(self.__userNum):
				RS, flag = self.__single_test(eachUser, CCF, userItemTestList, algorithmType, weight_THRESHOLD)
				aveRScore += RS
				tempUserNum += flag
			if tempUserNum != 0:
				aveRScore /= tempUserNum
			print "RACF"
			print aveRScore


		#test RA
		if algorithmType == self.__algoType_RA:
			aveRScore = 0.0#average ranking score
			tempUserNum = self.__userNum
			CCF.set_PesudoAM(1, Beta)
			for eachUser in range(self.__userNum):
				RS, flag = self.__single_test(eachUser, CCF, userItemTestList, algorithmType)
				aveRScore += RS
				tempUserNum += flag
			if tempUserNum != 0:
				aveRScore /= tempUserNum
			print "RA"
			print aveRScore
		
		return aveRScore


	def __single_test(self, objectUID, ccfHandler, userItemTestList, algorithmType, weight_THRESHOLD=[]):
		"""test for every user with criterions such as rank score, precision, recall"""
		if algorithmType == self.__algoType_RA:#RA
			sortedScoreDict = {}
			tmpList = list(ccfHandler.adjacentMatrix[:, objectUID])
			for index, score in enumerate(list(ccfHandler.RMatrix[:, objectUID])):
				if tmpList[index] != 1:#find items not collected yet
					sortedScoreDict.update(dict([(index, score)]))#to be improved, confusion may exist
			sortedScoreList = sorted(sortedScoreDict.iteritems(), key=lambda d:d[1],reverse = True)

		elif algorithmType == self.__algoType_CF or algorithmType == self.__algoType_RACF:
			sortedScoreList = ccfHandler.calc_CF(objectUID, algorithmType, weight_THRESHOLD)
		else:
			print "algorithm type error!"
			sys.exit(0)

		RS = 0.0#ranking score=L/N
		precision = 0.0#precision=TP/(TP+FP)
		recall = 0.0#recall=TP/(TP+FN)
		tmpN = len(sortedScoreList)#length of ordered score queue
		tmpItemTestNum = len(userItemTestList[objectUID])
		if tmpItemTestNum != 0:
			numOfCollectedItems_test = 0
			for item, rating in userItemTestList[objectUID].iteritems():
				if rating == 1:#find collected items in test dataset
					numOfCollectedItems_test += 1
					tmpItemIndex = self.__itemList.index(item)
					try:
						RS += (1.0 + sortedScoreList.index((tmpItemIndex, dict(sortedScoreList).get(tmpItemIndex))))/float(tmpN)
					except Exception, e:
						print e
						pdb.set_trace()
						sys.exit()
			if numOfCollectedItems_test != 0:
				RS /= numOfCollectedItems_test
				return [RS, 0]
			else:
				return [RS, -1]
		else:
			return [RS, -1]#-1 denotes that we can test nothing for this user




if __name__ == '__main__':
	t = testRACF()
	rst_ARS = t.nfold_cross_validation(10)

	algoType_CF = 0
	algoType_RA = 1
	algoType_RACF = 2
	n = range(10)
	plt.figure()
	plt.title("ARS for 3 different recommendation algorithms")
	plt.xlabel("10-fold")
	plt.ylabel("average ranking score")
	plt.plot(n,rst_ARS[algoType_CF],"g-",linestyle="--",label="CF: ARS")
	plt.plot(n,rst_ARS[algoType_RA],"r-",linestyle="-.",label="RA: ARS")
	plt.plot(n,rst_ARS[algoType_RACF],"b-",linestyle="-",label="RACF: ARS")
	plt.show()
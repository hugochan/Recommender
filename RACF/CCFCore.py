#!/usr/bin/env python
#encoding=utf-8

import numpy as np
import sys
import pdb

class CCFHandler(object):
	"""base on the article Content-Boosted Collaborative Filtering for Improved Recommendations written by Melville et al.
		we compensate the adjacentMatrix"""
	def __init__(self, adjacentMatrix):
		# try:
		# 	assert isinstance(N, int) == True
		# except:
		# 	print "N has a wrong data type !"
		# 	sys.exit()
		# else:
		# 	try:
		# 		assert N > 0
		# 	except:
		# 		print "N has a wrong assignment !"
		# 		sys.exit()
		try:
			assert isinstance(adjacentMatrix, np.ndarray) == True
		except:
			try:
				assert isinstance(adjacentMatrix, list) == True
			except:
				print "adjacentMatrix has a wrong data type !"
				sys.exit()
			else:
				self.adjacentMatrix = np.array(adjacentMatrix, dtype=np.float)
		else:
			self.adjacentMatrix = adjacentMatrix.astype(np.float)


		self.__rowNum = self.adjacentMatrix.shape[0]
		self.__colNum = self.adjacentMatrix.shape[1]
		# self.__topN = N

	def set_PesudoAM(self, flag, Beta=0):
		if flag == 1:
			from RACore import RAHandler
			RA = RAHandler(self.adjacentMatrix)
			self.RMatrix = RA.calc_RMatrix(Beta)
			fullOneMatrix = np.ones([self.__rowNum, self.__colNum])
			self.pesudoAdjacentMatrix = self.adjacentMatrix + (fullOneMatrix - self.adjacentMatrix)*self.RMatrix.round()
			# print "pesudoAdjacentMatrix"
			# print self.pesudoAdjacentMatrix
		elif flag == 0:
			self.pesudoAdjacentMatrix = self.adjacentMatrix
		else:
			print "pesudoAM mode error !"
			sys.exit()

	def calc_CF(self, objectUID, algorithmType, weight_THRESHOLD=[]):
		similiarity = self.__calc_similiarity(objectUID)
		if algorithmType == 0:
			scoreVector = self.__calc_score(similiarity)
		elif algorithmType == 2:
			scoreVector = self.__calc_score_boost(similiarity, objectUID, weight_THRESHOLD)
		else:
			print "algorithmType error!"
			sys.exit()

		sortedScoreList = self.__sort_score(scoreVector, objectUID)
		return sortedScoreList#return top N sorted scores

	def __calc_similiarity(self, objectUID):
		tmp = list(np.sum(self.pesudoAdjacentMatrix[:,objectUID]**2)*((self.pesudoAdjacentMatrix**2).sum(0)))
		while 0 in tmp:#solve the zero-division problem
			tmp[tmp.index(0)] = 1
		simVector = np.dot(np.transpose(self.pesudoAdjacentMatrix), self.pesudoAdjacentMatrix[:,objectUID])/np.array(tmp)
		# print self.__simVector.shape
		return simVector

	def __calc_score(self, similiarity):
		scoreVector = np.zeros([self.__rowNum, 1])
		tmp = np.sum(similiarity)
		if tmp == 0:#solve the zero-division problem
			tmp = 1
		scoreVector = np.dot(self.pesudoAdjacentMatrix, similiarity)/tmp#weighted average
		# print "scoreVector"
		return scoreVector

	def __calc_score_boost(self, similiarity, objectUID, weight_THRESHOLD):
		[hw, sw] = self.__calc_weight(objectUID, weight_THRESHOLD)
		# tmp = (hw*similiarity).sum(0)
		tmp = sw + (hw*similiarity).sum(0) - hw[objectUID]*similiarity[objectUID]

		if tmp == 0:#solve the zero-division problem
			tmp = 1
		# pdb.set_trace()
		# scoreVector_boost = np.dot(self.pesudoAdjacentMatrix, hw*self.__simVector)/tmp#weighted average
		scoreVector_boost = ((sw-hw[objectUID]*similiarity[objectUID])*self.pesudoAdjacentMatrix[:, objectUID]+np.dot(self.pesudoAdjacentMatrix, hw*similiarity))/tmp#weighted average
		return scoreVector_boost

	def __calc_weight(self, objectUID, weight_THRESHOLD):
		"""calculate some weight to boost the combination of algorithms"""
		# denominator_hm = np.ones([self.__colNum, ])*weight_THRESHOLD[0]

		# #1. calculate harmonic mean weighting 
		# n_object = np.sum(self.adjacentMatrix[:, objectUID])
		# m_objectUser = (n_object < weight_THRESHOLD[0]) and  n_object/weight_THRESHOLD[0] or 1.0
		# n_otherUsers = self.adjacentMatrix.sum(0)#shape: (self.__colNum, )
		# m_otherUsers = n_otherUsers/denominator_hm#shape: (self.__colNum, )
		# # print "n_otherUsers.shape"
		# # print n_otherUsers.shape
		# # print "m_otherUsers.shape"
		# # print m_otherUsers.shape
		# i = 0
		# for each in m_otherUsers:#performance to be improved
		# 	if each > 1:
		# 		m_otherUsers[i] = 1.0#the maximum weight is 1.0
		# 	i += 1
		# #harmonic mean weighting, shape: (self.__colNum, )
		# hm = 2*m_objectUser*m_otherUsers/(m_objectUser+m_otherUsers)
		hm = 0

		#2. calculate significance weighting
		#significance weighting, shape: (self.__colNum, )
		denominator_sg = np.ones([self.__colNum, ])*weight_THRESHOLD[1]
		sg = np.dot(np.transpose(self.adjacentMatrix), self.adjacentMatrix[:, objectUID])/denominator_sg
		i = 0
		for each in sg:#performance to be improved
			if each > 1:
				sg[i] = 1.0#the maximum weight is 1.0
			i += 1

		#3. calculate harmonic correlation weight
		hw = hm + sg
		# print "hm.shape"
		# print hm.shape

		#4. calculate self weighting
		# max_THRESHOLD = 1.0
		# # self weighting
		# if n_object < weight_THRESHOLD[2]:
		# 	print 1
		# sw = (n_object < weight_THRESHOLD[2]) and max_THRESHOLD*n_object/weight_THRESHOLD[2] or max_THRESHOLD
		sw = 0#sw = 0 is best
		return [hw, sw]

	def __sort_score(self, scoreVector, objectUID):
		"""sort and return all the items not collected yet as well as their scores"""
		sortedScoreDict = {}
		tmpList = list(self.adjacentMatrix[:,objectUID])
		for index, score in enumerate(list(scoreVector)):
			if tmpList[index] != 1:#find items not collected yet
				sortedScoreDict.update(dict([(index,score)]))#to be improved, confusion may exist
		sortedScoreList = sorted(sortedScoreDict.iteritems(), key=lambda d:d[1],reverse = True)
		# print d
		# print sortedScoreList
		return sortedScoreList


if __name__ == '__main__':
	CCF = CCFHandler([[1,1,0,1],[0,1,1,0],[0,1,1,1]], 0)
	CCF.set_PesudoAM(1)
	CCF.calc_similiarity()
	CCF.calc_score()
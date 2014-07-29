#!/usr/bin/env python
#encoding=utf-8

import numpy as np
import sys,time
import pdb

class RAHandler(object):
	"""base on the article Bipartite Network Projection and Personal Recommendation written by Zhou T et al."""
	def __init__(self, adjacentMatrix):
		"""standardize adjacentMatrix data type, the element ail links user l and item i
			oi and oj denotes item i and j, respectively """
		try:
			assert isinstance(adjacentMatrix, np.ndarray) == True
		except:
			try:
				assert isinstance(adjacentMatrix, list) == True
			except:
				print "adjacentMatrix has a wrong data type !"
				sys.exit()
			else:
				self.__adjacentMatrix = np.array(adjacentMatrix, dtype=np.float)
		else:
			self.__adjacentMatrix = adjacentMatrix.astype(np.float)

		self.__rowNum = self.__adjacentMatrix.shape[0]
		self.__colNum = self.__adjacentMatrix.shape[1]
		self.__get_RAMatrix()

	def __get_RAMatrix(self):
		"""genetate a Resourse-Allocation Matrix: W"""
		# t_start = time.clock()
		tmp = list(self.__adjacentMatrix.sum(0))
		while 0 in tmp:#solve the zero-division problem
			tmp[tmp.index(0)] = 1
		tmp2 = list(self.__adjacentMatrix.sum(1))
		while 0 in tmp2:#solve the zero-division problem
			tmp2[tmp2.index(0)] = 1
		#using matrix multiplication to replace double-loop
		self.__W = np.dot(self.__adjacentMatrix, np.transpose(self.__adjacentMatrix/np.array(tmp)))/np.array(tmp2)
		# t_finish = time.clock()
		# print "runing time in '__get_RAMatrix'"
		# print t_finish-t_start

	def calc_RVector(self, uID):
		"""calculate a final Resourse Vector for object user: F'"""
		fVector_init = self.__adjacentMatrix[:,uID]
		tmp = list(self.__W.sum(1))
		while 0 in tmp:#solve the zero-division problem
			tmp[tmp.index(0)] = 1
		fVector = np.dot(self.__W, fVector_init)/np.array(tmp)#weighted-average
		return fVector.round()

	def calc_RMatrix(self, Beta):
		"""calculate a final Resourse Matrix"""
		tmpWeightSum = list(self.__W.sum(1))
		while 0 in tmpWeightSum:
			tmpWeightSum[tmpWeightSum.index(0)] = 1
		weightSumMatrix = np.transpose(np.array([tmpWeightSum for j in np.arange(self.__colNum)]))

		self.__initSourceMatrix = self.__adjacentMatrix*(weightSumMatrix**Beta)#init source matrix by degrees of items 
		fMatrix = np.dot(self.__W, self.__initSourceMatrix)/weightSumMatrix#weighted-average
		return fMatrix#round

if __name__ == '__main__':
	RA = RAHandler([[1,1,0,1],[0,1,1,0],[0,1,1,1]])
	print RA.calc_RMatrix(0)
#!/usr/bin/env python
#encoding=utf-8

import sys, time, json
import scipy.sparse
import scipy.io
import redis.client
import pdb


class MassDiffusion(object):
    """docstring for MassDiffusion"""
    def __init__(self, filepath, dataset_name, split_traintest):
        super(MassDiffusion, self).__init__()
        # import redis.client
        self._filepath = filepath
        self.dataset_name = dataset_name
        self.split_traintest = split_traintest
        # self.userset = {}# user set: {"user":index,...}
        self.itemset = {}# item set: {"item":index,...}
        self.trainSet = {}# instance data set: {user:[item,...],...}
        self.testSet = {}
        self.instancenum = 0# num of instance
        # import redis
        # self.rds = redis.client.Redis(host="localhost", port=6379, db=0)


    def import_data(self, method="online"):
        filepath = "./offline_results/%s/"%self.dataset_name
        if method == "online":    
            try:
                with open(self._filepath, 'r') as f:
                    tmp_itemset = []# item set
                    temp_instanceSet = {}

                    templine = f.readline()
                    while(templine):
                        self.instancenum += 1
                        temp = templine.split(',')[:3]
                        user = int(temp[0])
                        item = int(temp[1])
                        time_stamp = int(temp[2][:-1])
                        tmp_itemset.append(item)
                        try:
                            temp_instanceSet[user].append([item, time_stamp])
                        except:
                            temp_instanceSet[user] = [[item, time_stamp]]
                        templine = f.readline()
            except Exception, e:
                print "import datas error !"
                print e
                sys.exit()
            f.close()


            temp_userset = temp_instanceSet.keys()
            tmp_itemset = list(set(tmp_itemset))# remove redundancy
            self.usernum = len(temp_userset)
            self.itemnum = len(tmp_itemset)

            # for user_index in range(self.usernum):
            #     self.userset[temp_userset[user_index]] = user_index
            for item_index in range(self.itemnum):
                self.itemset[tmp_itemset[item_index]] = item_index

                    
            # replace the key and value of instanceSet with user_index and item_index
            iterator = temp_instanceSet.iteritems()
            temp_instanceSet = {}
            uindex = 0
            for k, v in iterator:
                i = 0
                num = len(v)
                for eachrecord in sorted(v, key=lambda d:d[1],reverse = False):
                    if i != num - 1:
                        try:
                            self.trainSet[uindex].append(self.itemset[eachrecord[0]])
                        except:
                            self.trainSet[uindex] = [self.itemset[eachrecord[0]]]
                    else:
                        self.testSet[uindex] = [self.itemset[eachrecord[0]]]
                    i += 1
                uindex += 1


            try:
                self.store_data(json.dumps(self.trainSet), filepath + "trainset.json")
                self.store_data(json.dumps(self.testSet), filepath + "testset.json")
                self.store_data(json.dumps({"usernum":self.usernum, "itemnum":self.itemnum, "instancenum":self.instancenum}), filepath + "statistics.json")
            except Exception, e:
                print e
                sys.exit()

            print "user num: %s"%self.usernum
            print "item num: %s"%self.itemnum
            print "instance num: %s"%self.instancenum
        
        elif method == "offline":
            self.trainSet = self.read_data(filepath + "trainset.json")
            self.testSet = self.read_data(filepath + "testset.json")
            statistics = self.read_data(filepath + "statistics.json")
            if self.testSet != "" and self.trainSet != "" and statistics != "":
                try:
                    self.trainSet = json.loads(self.trainSet)
                    self.testSet = json.loads(self.testSet)
                    statistics = json.loads(statistics)
                except Exception, e:
                    print e
                    sys.exit()
                self.usernum = statistics["usernum"]
                self.itemnum = statistics["itemnum"]
                self.instancenum = statistics["instancenum"]
                print "user num: %s"%self.usernum
                print "item num: %s"%self.itemnum
                print "instance num: %s"%self.instancenum
            else:
                print "read nothing!"
                sys.exit()

        else:
            print "method arg error !"
            sys.exit()


    def create_ui_matrix(self, method="online"):
        filepath = "./offline_results/%s/ui_matrix"%self.dataset_name
        if method == "online":
            self.ui_matrix = scipy.sparse.lil_matrix((self.itemnum, self.usernum))
            iterator = self.trainSet.iteritems()
            self.trainSet = {}
            for user, item in iterator:
                for eachitem in item:
                    self.ui_matrix[eachitem, int(user)] = 1

            try:
                scipy.io.savemat(filepath, {"ui_matrix":self.ui_matrix}, oned_as='row')
            except Exception, e:
                print e
                sys.exit()
        
        elif method == "offline":
            self.trainSet = {}
            try:
                self.ui_matrix = scipy.io.loadmat(filepath, mat_dtype=False)["ui_matrix"]
            except Exception,e:
                print e
                sys.exit()
        else:
            print "method arg error !"
            sys.exit()
    
        self.ui_matrix = self.ui_matrix.tocsr()
        self.ui_matrix_sum0 = scipy.sparse.csr_matrix(self.ui_matrix.sum(0))
        self.ui_matrix_sum1 = scipy.sparse.csr_matrix(self.ui_matrix.sum(1))



    def compute(self, n):
        """test"""
        print "redis"
        self.rds = redis.client.Redis(host="localhost", port=6379, db=0)
        print self.rds.get('f')
        print "sleep %s s..."%n
        time.sleep(n)
        a = scipy.sparse.lil_matrix((3,4))
        print "done !"
        return n


    def calc_RAMatrix(self, scope, groupid):
            """genetate a Resourse-Allocation Matrix: W"""
            # import redis.client
            self.rds = redis.client.Redis(host="192.168.1.106", port=6379, db=0)

            for eachitem in range(scope[0], scope[1]):
                temp_w = (self.ui_matrix.dot((self.ui_matrix[eachitem, :]/self.ui_matrix_sum0).transpose())/self.ui_matrix_sum1).transpose()
                data = json.dumps(temp_w.toarray().tolist())
                temp_w = 0
                
                try:
                    self.rds.hset(groupid, eachitem, data)
                    data = 0
                except Exception, e:
                    print e
                    sys.exit()

                if eachitem % 100 == 0:
                    print "key:%s,field:%s"%(groupid, eachitem)

    
    def calc_RVector(self, uID):
        """calculate a final Resourse Vector for object user: F'"""
        fVector_init = self.ui_matrix[:, uID]
        # tmp = list(self.__W.sum(1))

        block_size = 10000
        fVector = {}
        for eachitem in range(self.itemnum):
            if fVector_init[eachitem, 0] == 0:
                fVector[eachitem] = scipy.sparse.csc_matrix(json.loads(self.rds.hget(int(eachitem/block_size), eachitem))).dot(fVector_init)
        return fVector

    def single_test(self, uID):
        fVector = self.calc_RVector(uID)
        fVector =  sorted(fVector.iteritems(), key=lambda d:d[1], reverse = True)
        single_rankingscore = (fVector.index((self.testSet[uID][0], dict(fVector)[self.testSet[uID][0]])) + 1)/len(fVector)
        return single_rankingscore

    def test(self):
        total_RS = 0
        for user in self.testSet.keys():
            total_RS += self.single_test(int(user))
        average_RS = total_RS/self.usernum
        print "average ranking score: %s"%average_RS
        return average_RS

    def store_data(self, data, filepath):
        try:
            with open(filepath, 'w') as f:
                f.write(data)
        except Exception, e:
            print "store datas error !"
            print e
            return -1
        f.close()
        return 0

    def read_data(self, filepath):
        try:
            with open(filepath, 'r') as f:
                data = f.read()
        except Exception, e:
            print "read datas error !"
            print e
            data = ""
        f.close()
        return data



if __name__ == '__main__':
    md = MassDiffusion(filepath="../../../data/shujuchuli_0618/fengniao_chengyu_0618.txt", dataset_name="fengniao", split_traintest="yes")
    t0 = time.clock()
    md.import_data(method="offline")
    t1 = time.clock()
    print "import_data costs: %ss"%(t1-t0)

    t0 = time.clock()
    md.create_ui_matrix(method="offline")
    t1 = time.clock()
    print "create_ui_matrix costs: %ss"%(t1-t0)
 
    t0 = time.clock()
    md.calc_RAMatrix()
    t1 = time.clock()
    print "calc_RAMatrix costs: %ss"%(t1-t0)

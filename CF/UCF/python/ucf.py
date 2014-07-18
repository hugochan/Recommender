#!/usr/bin/env python
#encoding=utf-8

import sys, time, json
import scipy.sparse
import numpy
import random
import pdb

class ucf(object):
    """user-based collaborative filtering"""
    def __init__(self, filepath, dataset_name, split_trainprobe):
        super(ucf, self).__init__()
        self._filepath = filepath
        self.dataset_name = dataset_name
        self.split_trainprobe = split_trainprobe
        self.userset = {}# user set: {"user":index,...}
        self.itemset = {}# item set: {"item":index,...}
        self.trainset = {}# instance data set: {user:[item,...],...}
        self.probeset = {}

    def import_datas(self, method):
        filepath = "../offline_results/%s/ucf/"%self.dataset_name
        if method == "online":
            if self.split_trainprobe == "yes": 
                try:
                    with open(self._filepath+"u.data", 'r') as f:
                        tmp_itemset = []# item set
                        temp_instanceset = {}
                        instancenum = 0
                        templine = f.readline()
                        while(templine):
                            instancenum += 1
                            temp = templine.split('\t')[:4]
                            user = int(temp[0])
                            item = int(temp[1])
                            time_stamp = int(temp[3])
                            tmp_itemset.append(item)
                            if int(temp[2]) >= 3:
                                try:
                                    temp_instanceset[user].append([item, time_stamp])
                                except:
                                    temp_instanceset[user] = [[item, time_stamp]]
                            templine = f.readline()
                except Exception, e:
                    print "import datas error !"
                    print e
                    sys.exit()
                f.close()

                temp_userset = temp_instanceset.keys()
                tmp_itemset = list(set(tmp_itemset))# remove redundancy
                self.usernum = len(temp_userset)
                self.itemnum = len(tmp_itemset)

                # re-mapping uid & iid, form like this {username:uid, ...} & {itemname:iid, ...} 
                for uid in range(self.usernum):
                    self.userset[temp_userset[uid]] = uid
                for iid in range(self.itemnum):
                    self.itemset[tmp_itemset[iid]] = iid

                # replace the username and itemname of instanceset with uid and iid
                iterator = temp_instanceset.iteritems()
                temp_instanceset = {}# release the space
                uid = 0
                for user, records in iterator:
                    i = 0
                    recordnum = len(records)
                    for eachrecord in sorted(records, key=lambda d:d[1],reverse = False):
                        if i != recordnum - 1:
                            try:
                                self.trainset[uid].append(self.itemset[eachrecord[0]])
                            except:
                                self.trainset[uid] = [self.itemset[eachrecord[0]]]
                        else:
                            self.probeset[uid] = [self.itemset[eachrecord[0]]]
                        i += 1
                    uid += 1

                try:
                    self.store_data(json.dumps(self.trainset), filepath + "trainset.json")
                    self.store_data(json.dumps(self.probeset), filepath + "probeset.json")
                    self.store_data(json.dumps({"usernum":self.usernum, "itemnum":self.itemnum, "instancenum":instancenum}), filepath + "statistics.json")
                except Exception, e:
                    print e
                    sys.exit()

                print "user num: %s"%self.usernum
                print "item num: %s"%self.itemnum
                print "instance num: %s"%instancenum

            elif self.split_trainprobe == "no":
                # read train datas
                try:
                    with open(self._filepath+"10000samples.txt", 'r') as f:
                        tmp_train_itemset = []# item set
                        train_instanceset = {}
                        train_instancenum = 0

                        templine = f.readline()
                        while(templine):
                            train_instancenum += 1
                            temp = templine.split('\t')[:2]
                            user = int(temp[0])
                            item = int(temp[1])
                            tmp_train_itemset.append(item)
                            try:
                                train_instanceset[user].append(item)
                            except:
                                train_instanceset[user] = [item]
                            templine = f.readline()
                except Exception, e:
                    print "import datas error !"
                    print e
                    sys.exit()
                f.close()

                # read test datas
                try:
                    with open(self._filepath+"test10000samples.txt", 'r') as f:
                        tmp_test_itemset = []# item set
                        test_instanceset = {}
                        test_instancenum = 0

                        templine = f.readline()
                        while(templine):
                            test_instancenum += 1
                            temp = templine.split('\t')[:2]
                            user = int(temp[0])
                            item = int(temp[1])
                            tmp_test_itemset.append(item)
                            try:
                                test_instanceset[user].append(item)
                            except:
                                test_instanceset[user] = [item]
                            templine = f.readline()
                except Exception, e:
                    print "import datas error !"
                    print e
                    sys.exit()
                f.close()

                temp_userset = train_instanceset.keys()
                tmp_itemset = list(set(tmp_train_itemset))# remove redundancy
                tmp_itemset_add = list(set(tmp_test_itemset) - set(tmp_itemset))
                self.usernum = len(temp_userset)
                self.itemnum = len(tmp_itemset)
                self.itemnum_add = len(tmp_itemset_add)

                for user_index in range(self.usernum):
                    self.userset[temp_userset[user_index]] = user_index
                for item_index in range(self.itemnum):
                    self.itemset[tmp_itemset[item_index]] = item_index
                for item_index in range(self.itemnum_add):
                    self.itemset[tmp_itemset_add[item_index]] = item_index + self.itemnum
                        
                # replace the key and value of train_instanceset with user_index and item_index
                iterator = train_instanceset.iteritems()
                train_instanceset = {}
                uindex = 0
                for k, v in iterator:
                    for eachitem in v:
                        try:
                            self.trainset[uindex].append(self.itemset[eachitem])
                        except:
                            self.trainset[uindex] = [self.itemset[eachitem]]
                    uindex += 1

                count = 0
                # replace the key and value of test_instanceset with user_index and item_index
                iterator = test_instanceset.iteritems()
                test_instanceset = {}
                for k, v in iterator:
                    for eachitem in v:
                        self.probeset[self.userset[k]] = [self.itemset[eachitem]]
                        if self.itemset[eachitem] in self.trainset[self.userset[k]]:
                            count+=1
                print "count %s"%count

                # store
                try:
                    self.store_data(json.dumps(self.trainset), filepath + "trainset.json")
                    self.store_data(json.dumps(self.probeset), filepath + "probeset.json")
                    self.store_data(json.dumps({"usernum":self.usernum, "train itemnum":self.itemnum,\
                        "test itemnum":self.itemnum_add, "train instancenum":train_instancenum, \
                            "test instancenum":test_instancenum}), filepath + "statistics.json")
                except Exception, e:
                    print e
                    sys.exit()

                print "user num: %s"%self.usernum
                print "trainset item num: %s"%self.itemnum
                print "testset item added num: %s"%self.itemnum_add
                print "trainset instance num: %s"%train_instancenum
                print "testset instance num: %s"%test_instancenum

        else:
            print "split_trainprobe arg error !"
            sys.exit()

    def create_ui_matrix(self, method):
        if method == "online":
            self.ui_matrix = scipy.sparse.lil_matrix((self.itemnum, self.usernum))
            for user, items in self.trainset.iteritems():
                for eachitem in items:
                    self.ui_matrix[eachitem, int(user)] = 1.0
            self.ui_matrix = self.ui_matrix.tocsc()


    def calc_usersimilarity(self, uid):
        """cosine similiarity"""
        usersimilarity = self.ui_matrix[:, uid].transpose().dot(self.ui_matrix)/\
            scipy.sparse.csc_matrix(numpy.array(self.ui_matrix[:, uid].sum(0)[0, 0]*\
                self.ui_matrix.sum(0))**0.5)
        return usersimilarity

    def calc_single_recommendscore(self, uid):
        recommendscore = []
        usersimilarity = self.calc_usersimilarity(uid)
        score = (usersimilarity.dot(self.ui_matrix.transpose())/(usersimilarity.sum(1)[0, 0])).toarray().tolist()[0]
        iid = 0
        for eachscore in score:
            recommendscore.append((iid, eachscore))
            iid += 1
        return recommendscore

    def recommend(self, scope, groupid):
        filepath = "../offline_results/%s/ucf/"%self.dataset_name
        user_recommendscore = {}
        for user in range(scope[0], scope[1]):
            user_recommendscore[user] = self.calc_single_recommendscore(user)

        try:
            with open(filepath+"temp/"+"ucf-user_recommendscore.txt_%s"%groupid, "w") as f:
                for user, recommendscore in user_recommendscore.iteritems():
                    data = []
                    for item, score in recommendscore:
                        data.append("%s    %s    %s\n"%(user, item, score)) 
                    f.writelines(data)
        except Exception, e:
            print e
            print "store user:%s - user:%s recommend scores error !"%(scope[0], scope[1])
            sys.exit()
        
        f.close()
        return 0

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
    ucf = ucf(filepath="../../../../../data/public_datas/movielens-100k/ml-100k/", dataset_name="ml-100k", split_trainprobe="yes")    
    ucf.import_datas(method="online")
    ucf.create_ui_matrix(method="online")
    ucf.recommend((0, 20), 0)
#!/usr/bin/env python
#encoding=utf-8

import sys, time, json
import scipy.sparse
import scipy.io
import random
import pdb


class HeatConduction(object):
    """docstring for HeatConduction"""
    def __init__(self, filepath, dataset_name, split_trainprobe):
        super(HeatConduction, self).__init__()
        self._filepath = filepath
        self.dataset_name = dataset_name
        self.split_trainprobe = split_trainprobe
        self.userset = {}# user set: {"user":index,...}
        self.itemset = {}# item set: {"item":index,...}
        self.trainset = {}# instance data set: {user:[item,...],...}
        self.probeset = {}


    def import_datas(self, method="online"):
        filepath = "../offline_results/%s/hc/"%self.dataset_name
        if method == "online":
            if self.split_trainprobe == "yes":
                try:
                    with open(self._filepath+"caixin_sampling_0.2.txt", 'r') as f:
                        tmp_itemset = []# item set
                        temp_instanceset = {}
                        instancenum = 0
                        templine = f.readline()
                        while(templine):
                            instancenum += 1
                            # temp = templine.split(' ')[:4]
                            temp = templine.split('    ')[:3]

                            user = int(temp[0])
                            item = int(temp[1])
                            # time_stamp = int(temp[3])
                            time_stamp = int(temp[2][:-1])

                            tmp_itemset.append(item)
                            # if int(temp[2]) >= 3:
                            #     try:
                            #         temp_instanceset[user].append([item, time_stamp])
                            #     except:
                            #         temp_instanceset[user] = [[item, time_stamp]]
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


    def create_ui_matrix(self, method="online"):
        filepath = "../offline_results/%s/ui_matrix"%self.dataset_name
        if method == "online":
            self.ui_matrix = scipy.sparse.lil_matrix((self.itemnum, self.usernum))
            iterator = self.trainset.iteritems()
            for user, item in iterator:
                for eachitem in item:
                    self.ui_matrix[eachitem, int(user)] = 1

            try:
                scipy.io.savemat(filepath, {"ui_matrix":self.ui_matrix}, oned_as='row')
            except Exception, e:
                print e
                sys.exit()
        
        elif method == "offline":
            try:
                self.ui_matrix = scipy.io.loadmat(filepath, mat_dtype=False)["ui_matrix"]
            except Exception,e:
                print e
                sys.exit()
        else:
            print "method arg error !"
            sys.exit()
    
        self.ui_matrix = self.ui_matrix.tocsr()
        self.userdegree = scipy.sparse.csr_matrix(self.ui_matrix.sum(0))
        self.itemdegree = scipy.sparse.csr_matrix(self.ui_matrix.sum(1))
        # self.RAMatrix = scipy.sparse.lil_matrix((self.itemnum, self.itemnum))


    def calc_RAMatrix(self, iid):
        """genetate a Resourse-Allocation Matrix: each item allocates resourses to iid"""
        # shape: (1, itemnum)
        w = (self.ui_matrix.dot((self.ui_matrix[iid, :]/self.userdegree).transpose())/self.itemdegree[iid, 0]).transpose()
        return w


    def calc_single_recommendscore(self, iid):
        """calc recommend scores which might be valued by all users for item iid"""
        w = self.calc_RAMatrix(iid)
        score = w.dot(self.ui_matrix)
        recommendscore = zip(range(self.usernum), score.toarray().tolist()[0])
        return recommendscore

    def recommend(self, scope, groupid):
        filepath = "../offline_results/%s/hc/"%self.dataset_name
        item_recommendscore = {}
        for item in range(scope[0], scope[1]):
            item_recommendscore[item] = self.calc_single_recommendscore(item)

        try:
            with open(filepath+"temp/"+"hc-item_recommendscore.txt_%s"%groupid, "w") as f:
                for item, recommendscore in item_recommendscore.iteritems():
                    data = []
                    for user, score in recommendscore:
                        data.append("%s    %s    %s\n"%(user, item, score)) 
                    f.writelines(data)
        except Exception, e:
            print e
            print "store item:%s - item:%s recommend scores error !"%(scope[0], scope[1])
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
    md = HeatConduction(filepath="../../../../data/daduchouyangshuju1/xicichouyang/", dataset_name="xici", split_trainprobe="no")
    t0 = time.clock()
    md.import_datas(method="online")
    t1 = time.clock()
    print "import_datas costs: %ss"%(t1-t0)

    t0 = time.clock()
    md.create_ui_matrix(method="online")
    t1 = time.clock()
    print "create_ui_matrix costs: %ss"%(t1-t0)

    t0 = time.clock()
    md.recommend((0, 3), 0)
    t1 = time.clock()
    print "predict costs: %ss"%(t1-t0)
#!/usr/bin/env python
#encoding=utf-8

import sys, time, json
import scipy.sparse
import scipy.io
import numpy
import random
import pdb


class TraHeatConduction(object):
    """docstring for TraHeatConduction"""
    def __init__(self, filepath, dataset_name, split_trainprobe, decay_factor):
        super(TraHeatConduction, self).__init__()
        self._filepath = filepath
        self.dataset_name = dataset_name
        self.split_trainprobe = split_trainprobe
        self.decay_factor = decay_factor
        self.userset = {}# user set: {"user":index,...}
        self.itemset = {}# item set: {"item":index,...}
        self.trainset = {}# instance data set: {user:[item,...],...}
        self.probeset = {}


    def import_datas(self, method="online"):
        filepath = "../offline_results/%s/tra_hc/"%self.dataset_name
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
                            temp = templine.split('\t')[:3]
                            user = int(temp[0])
                            item = int(temp[1])
                            time_stamp = int(temp[2][:-1])
                            tmp_train_itemset.append(item)
                            try:
                                train_instanceset[user].append([item, time_stamp])
                            except:
                                train_instanceset[user] = [[item, time_stamp]]
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
                            temp = templine.split('\t')[:3]
                            user = int(temp[0])
                            item = int(temp[1])
                            time_stamp = int(temp[2][:-1])
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
                for user, records in iterator:
                    for eachrecord in records:
                        try:
                            self.trainset[uindex].append([self.itemset[eachrecord[0]], eachrecord[1]])
                        except:
                            self.trainset[uindex] = [[self.itemset[eachrecord[0]], eachrecord[1]]]
                    uindex += 1

                count = 0
                # replace the key and value of test_instanceset with user_index and item_index
                iterator = test_instanceset.iteritems()
                test_instanceset = {}
                for user, items in iterator:
                    for eachitem in items:
                        self.probeset[self.userset[user]] = [self.itemset[eachitem]]
                        if self.itemset[eachitem] in dict(self.trainset[self.userset[user]]).keys():
                            count += 1
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


    def create_uit_matrix(self, method):
        filepath = "../offline_results/%s/tra_hc/"%self.dataset_name
        if method == "online":
            self.ui_matrix = scipy.sparse.lil_matrix((self.itemnum, self.usernum))
            self.time_matrix = scipy.sparse.lil_matrix((self.itemnum, self.usernum))
            for user, records in self.trainset.iteritems():
                for eachrecord in records:
                    self.ui_matrix[eachrecord[0], int(user)] = 1.0
                    self.time_matrix[eachrecord[0], int(user)] = eachrecord[1]
            self.ui_matrix = self.ui_matrix.tocsc()
            self.time_matrix = self.time_matrix.tocsc()
            self.tbased_resource = self.calc_tbased_resource(self.decay_factor)
            self.userdegree = scipy.sparse.csr_matrix(self.ui_matrix.sum(0))
            self.itemdegree = scipy.sparse.csr_matrix(self.ui_matrix.sum(1))
            # try:
            #     scipy.io.savemat(filepath+"uit_matrix", {"ui_matrix":self.ui_matrix, "time_matrix":self.time_matrix, "tbased_resource":self.tbased_resource}, oned_as='row')
            # except Exception, e:
            #     print e
            #     sys.exit()
        elif method == "offline":
            try:
                matrix = scipy.io.loadmat(filepath+"uit_matrix", mat_dtype=False)
                self.ui_matrix = matrix["ui_matrix"]
                self.time_matrix = matrix["time_matrix"]
                self.tbased_resource = matrix["tbased_resource"]                
            except Exception,e:
                print e
                sys.exit()
        else:
            print "method arg error !"
            sys.exit()


    def calc_RAMatrix(self, iid):
        """genetate a Resourse-Allocation Matrix: each item allocates resourses to iid"""
        # shape: (1, itemnum)
        w = (self.ui_matrix.dot((self.ui_matrix[iid, :]/self.userdegree).transpose())/self.itemdegree[iid, 0]).transpose()
        return w

    def calc_tbased_resource(self, decay_factor):
        memory = 0.5*1024*1024*1024
        block_size = int(memory/(self.usernum*8))
        block_num = int(self.itemnum/block_size)
        for block_id in range(block_num):
            item_time = self.time_matrix[block_id*block_size:(block_id+1)*block_size, :].todense()
            item_time = item_time.max(1) + 1 - item_time
            if block_id == 0:
                tbased_resource = scipy.sparse.csc_matrix(numpy.array(item_time)**decay_factor).multiply(self.ui_matrix[block_id*block_size:(block_id+1)*block_size, :])
            else:
                tbased_resource = scipy.sparse.vstack([tbased_resource, scipy.sparse.csc_matrix(numpy.array(item_time)**decay_factor).multiply(self.ui_matrix[block_id*block_size:(block_id+1)*block_size, :])], "csc")
        if self.itemnum - block_num*block_size != 0:
            item_time = self.time_matrix[block_num*block_size:self.itemnum, :].todense()
            item_time = item_time.max(1) + 1 - item_time
            tbased_resource = scipy.sparse.vstack([tbased_resource, scipy.sparse.csc_matrix(numpy.array(item_time)**decay_factor).multiply(self.ui_matrix[block_num*block_size:self.itemnum, :])], "csc")
        return tbased_resource

    def calc_single_recommendscore(self, iid):
        """calc recommend scores which might be valued by all users for item iid"""
        w = self.calc_RAMatrix(iid)
        score = w.dot(self.tbased_resource)
        recommendscore = zip(range(self.usernum), score.toarray().tolist()[0])
        return recommendscore

    def recommend(self, scope, groupid):
        filepath = "../offline_results/%s/tra_hc/"%self.dataset_name
        item_recommendscore = {}
        for item in range(scope[0], scope[1]):
            item_recommendscore[item] = self.calc_single_recommendscore(item)

        try:
            with open(filepath+"temp/"+"tra_hc-item_recommendscore.txt_%s"%groupid, "w") as f:
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
    md = TraHeatConduction(filepath="../../../../data/daduchouyangshuju1/xicichouyang/", dataset_name="xici", split_trainprobe="no", decay_factor=-0.4)
    t0 = time.clock()
    md.import_datas(method="online")
    t1 = time.clock()
    print "import_datas costs: %ss"%(t1-t0)

    t0 = time.clock()
    md.create_uit_matrix(method="online")
    t1 = time.clock()
    print "create_uit_matrix costs: %ss"%(t1-t0)

    t0 = time.clock()
    md.recommend((0, 1), 0)
    t1 = time.clock()
    print "predict costs: %ss"%(t1-t0)
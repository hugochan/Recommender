#!/usr/bin/env python
#encoding=utf-8

import sys, time, json
import scipy.sparse
import numpy
import random
import pdb

class tsn_ucf(object):
    """user-based collaborative filtering"""
    def __init__(self, filepath, dataset_name, split_trainprobe, decay_factor=1):
        super(tsn_ucf, self).__init__()
        self._filepath = filepath
        self.dataset_name = dataset_name
        self.split_trainprobe = split_trainprobe
        self.decay_factor = decay_factor
        self.userset = {}# user set: {"user":index,...}
        self.itemset = {}# item set: {"item":index,...}
        self.trainset = {}# instance data set: {user:[item,...],...}
        self.probeset = {}
        print "decay_factor: %s"%self.decay_factor

    def import_datas(self, method):
        if method == "online":
            filepath = "../offline_results/%s/tsn_ucf/"%self.dataset_name
            if self.split_trainprobe == "yes": 
                try:
                    with open(self._filepath+"caixin_sampling_0.2.txt", 'r') as f:
                        tmp_itemset = []# item set
                        temp_instanceset = {}
                        instancenum = 0
                        templine = f.readline()
                        while(templine):
                            instancenum += 1
                            # temp = templine.split('\t')[:4]
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
                                self.trainset[uid].append([self.itemset[eachrecord[0]], eachrecord[1]])
                            except:
                                self.trainset[uid] = [[self.itemset[eachrecord[0]], eachrecord[1]]]
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
                            time_stamp = int(temp[2])
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
                            time_stamp = int(temp[2])
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
        if method == "online":
            self.ui_matrix = scipy.sparse.lil_matrix((self.itemnum, self.usernum))
            self.time_matrix = scipy.sparse.lil_matrix((self.itemnum, self.usernum))
            for user, records in self.trainset.iteritems():
                for eachrecord in records:
                    self.ui_matrix[eachrecord[0], int(user)] = 1.0
                    self.time_matrix[eachrecord[0], int(user)] = eachrecord[1]
            self.ui_matrix = self.ui_matrix.tocsc()
            self.time_matrix = self.time_matrix.tocsc()

    def calc_usersimilarity(self, uid):
        """time-based cosine similiarity"""
        usersimilarity = self.ui_matrix[:, uid].transpose().dot(self.ui_matrix)/\
            scipy.sparse.csc_matrix(numpy.array(self.ui_matrix[:, uid].sum(0)[0, 0]*\
                self.ui_matrix.sum(0))**0.5)
    
        # relative time
        decay = []
        for uid2 in range(self.usernum):
            if uid2 == uid:
                decay.append(1)
            else:
                mask = self.ui_matrix[:, uid].multiply(self.ui_matrix[:, uid2])
                common_itemnum = int(mask.sum(0)[0, 0])
                if common_itemnum == 0:
                    decay.append(1)
                else:
                    uid_itdict = dict(zip(range(self.itemnum), self.time_matrix[:, uid].transpose().toarray().tolist()[0]))
                    uid2_itdict = dict(zip(range(self.itemnum), self.time_matrix[:, uid2].transpose().toarray().tolist()[0]))

                    iid = 0
                    for each in mask.transpose().toarray()[0]:
                        if each == 0:
                            try:
                                del uid_itdict[iid]
                                del uid2_itdict[iid]
                            except Exception, e:
                                print e
                                sys.exit()
                        iid += 1

                    uid_itemtime = sorted(uid_itdict.iteritems(), key=lambda d:d[1], reverse=False)
                    uid2_itemtime = sorted(uid2_itdict.iteritems(), key=lambda d:d[1], reverse=False)

                    for relative_time in range(common_itemnum):
                        uid_itemtime[relative_time] = (uid_itemtime[relative_time][0], relative_time)
                        uid2_itemtime[relative_time] = (uid2_itemtime[relative_time][0], relative_time)
                    uid_itemtime = dict(uid_itemtime)
                    uid2_itemtime = dict(uid2_itemtime)


                    absolute_difference = 0.0
                    for item, relative_time in uid_itemtime.iteritems():
                        absolute_difference += abs(relative_time-uid2_itemtime[item])

                    tmp_decay = (1+absolute_difference/common_itemnum)**(-1*self.decay_factor)
                    # if absolute_difference == 0:
                        # tmp_decay = 1
                    # else:
                        # tmp_decay = (absolute_difference/common_itemnum)**(-1*self.decay_factor)
                    
                    decay.append(tmp_decay)
        decay = scipy.sparse.csc_matrix(decay)
        usersimilarity = usersimilarity.multiply(decay)

        return usersimilarity

    def calc_single_recommendscore(self, uid):
        usersimilarity = self.calc_usersimilarity(uid)
        total_weight = usersimilarity.sum(1)[0, 0]
        if total_weight == 0:
            total_weight = 1
        score = (usersimilarity.dot(self.ui_matrix.transpose())/total_weight).toarray().tolist()[0]
        recommendscore = zip(range(self.itemnum), score)
        return recommendscore

    def recommend(self, scope, groupid):
        filepath = "../offline_results/%s/tsn_ucf/"%self.dataset_name
        user_recommendscore = {}
        for user in range(scope[0], scope[1]):
            user_recommendscore[user] = self.calc_single_recommendscore(user)

        try:
            with open(filepath+"temp/"+"tsn_ucf-user_recommendscore.txt_%s"%groupid, "w") as f:
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
    tsn = tsn_ucf(filepath="../../../../../data/caixin/", dataset_name="caixin", split_trainprobe="yes")    
    tsn.import_datas(method="online")
    tsn.create_uit_matrix(method="online")
    t0=time.clock()
    tsn.recommend((0, 1), 0)
    t1=time.clock()
    print "costs %ss"%(t1-t0)
#!/usr/bin/env python
#encoding=utf-8

import sys, time, json
import scipy.sparse
import scipy.io
import redis.client
import random
import pdb


class MassDiffusion(object):
    """docstring for MassDiffusion"""
    def __init__(self, filepath, dataset_name, split_traintest):
        super(MassDiffusion, self).__init__()
        # import redis.client
        self._filepath = filepath
        self.dataset_name = dataset_name
        self.split_traintest = split_traintest
        self.userset = {}# user set: {"user":index,...}
        self.itemset = {}# item set: {"item":index,...}
        self.trainSet = {}# instance data set: {user:[item,...],...}
        self.testSet = {}


    def import_data(self, method="online"):
        filepath = "../offline_results/%s/"%self.dataset_name
        if method == "online":
            if self.split_traintest == "yes": 
                try:
                    with open(self._filepath, 'r') as f:
                        tmp_itemset = []# item set
                        temp_instanceSet = {}
                        instancenum = 0

                        templine = f.readline()
                        while(templine):
                            instancenum += 1
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
                    self.store_data(json.dumps({"usernum":self.usernum, "itemnum":self.itemnum, "instancenum":instancenum}), filepath + "statistics.json")
                except Exception, e:
                    print e
                    sys.exit()

                print "user num: %s"%self.usernum
                print "item num: %s"%self.itemnum
                print "instance num: %s"%instancenum

            elif self.split_traintest == "no":
                # read train datas
                try:
                    with open(self._filepath+"10000Large20samples.txt", 'r') as f:
                        tmp_train_itemset = []# item set
                        train_instanceSet = {}
                        train_instancenum = 0

                        templine = f.readline()
                        while(templine):
                            train_instancenum += 1
                            temp = templine.split('\t')[:2]
                            user = int(temp[0])
                            item = int(temp[1])
                            tmp_train_itemset.append(item)
                            try:
                                train_instanceSet[user].append(item)
                            except:
                                train_instanceSet[user] = [item]
                            templine = f.readline()
                except Exception, e:
                    print "import datas error !"
                    print e
                    sys.exit()
                f.close()

                # read test datas
                try:
                    with open(self._filepath+"test10000Large20samples.txt", 'r') as f:
                        tmp_test_itemset = []# item set
                        test_instanceSet = {}
                        test_instancenum = 0

                        templine = f.readline()
                        while(templine):
                            test_instancenum += 1
                            temp = templine.split('\t')[:2]
                            user = int(temp[0])
                            item = int(temp[1])
                            tmp_test_itemset.append(item)
                            try:
                                test_instanceSet[user].append(item)
                            except:
                                test_instanceSet[user] = [item]
                            templine = f.readline()
                except Exception, e:
                    print "import datas error !"
                    print e
                    sys.exit()
                f.close()

                temp_userset = train_instanceSet.keys()
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
                        
                # replace the key and value of train_instanceSet with user_index and item_index
                iterator = train_instanceSet.iteritems()
                train_instanceSet = {}
                uindex = 0
                for k, v in iterator:
                    for eachitem in v:
                        try:
                            self.trainSet[uindex].append(self.itemset[eachitem])
                        except:
                            self.trainSet[uindex] = [self.itemset[eachitem]]
                    uindex += 1

                count = 0
                # replace the key and value of test_instanceSet with user_index and item_index
                iterator = test_instanceSet.iteritems()
                test_instanceSet = {}
                for k, v in iterator:
                    for eachitem in v:
                        self.testSet[self.userset[k]] = [self.itemset[eachitem]]
                        if self.itemset[eachitem] in self.trainSet[self.userset[k]]:
                            count+=1
                print "count %s"%count

                # store
                try:
                    self.store_data(json.dumps(self.trainSet), filepath + "trainset.json")
                    self.store_data(json.dumps(self.testSet), filepath + "testset.json")
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
                print "split_traintest arg error !"
                sys.exit()
        
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
                if self.split_traintest == "yes":
                    instancenum = statistics["instancenum"]
                    self.itemnum = statistics["itemnum"]
                    print "user num: %s"%self.usernum
                    print "item num: %s"%self.itemnum
                    print "instance num: %s"%instancenum
                elif self.split_traintest == "no":
                    self.itemnum = statistics["train itemnum"]
                    self.itemnum_add = statistics["test itemnum"]
                    train_instancenum = statistics["train instancenum"]
                    test_instancenum = statistics["test instancenum"]
                    print "user num: %s"%self.usernum
                    print "trainset item num: %s"%self.itemnum
                    print "testset item added num: %s"%self.itemnum_add
                    print "trainset instance num: %s"%train_instancenum
                    print "testset instance num: %s"%test_instancenum

            else:
                print "read nothing!"
                sys.exit()

        else:
            print "method arg error !"
            sys.exit()


    def create_ui_matrix(self, method="online"):
        filepath = "../offline_results/%s/ui_matrix"%self.dataset_name
        if method == "online":
            self.ui_matrix = scipy.sparse.lil_matrix((self.itemnum, self.usernum))
            iterator = self.trainSet.iteritems()
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


    def calc_RAMatrix(self, scope, groupid):
        """genetate a Resourse-Allocation Matrix: W"""
        # pdb.set_trace()
        try:
            if self.dataset_name == "caixin":
                rds_ram = redis.client.StrictRedis(host="localhost", port=6379, db=0)#ra matrix
            elif self.dataset_name == "fengniao":
                rds_ram = redis.client.StrictRedis(host="localhost", port=6380, db=0)#ra matrix
            elif self.dataset_name == "xici":
                rds_ram = redis.client.StrictRedis(host="localhost", port=6381, db=0)#ra matrix
            else:
                print "dataset name arg error !"
                sys.exit()
        except Exception, e:
            print e
            sys.exit()

        for eachitem in range(scope[0], scope[1]):
            temp_w = (self.ui_matrix.dot((self.ui_matrix[eachitem, :]/self.userdegree).transpose())/self.itemdegree[eachitem, 0]).transpose()
            data = json.dumps(temp_w.toarray().tolist())
            temp_w = 0

            try:
                rds_ram.hset(groupid, eachitem, data)
            except Exception, e:
                print e

    
    def calc_RVector(self, uID, rds_ram_handler, groupid):
        """calculate a final Resourse Vector for object user: F'"""
        fVector_init = self.ui_matrix[:, uID]
        fVector_final = scipy.sparse.csc_matrix((1, self.itemnum))
        # tmp = list(self.__W.sum(1))

        block_size = 100
        iteritems = range(self.itemnum)
        iteritems = (uID%2 == 0) and sorted(iteritems, reverse=True) or iteritems
        for eachitem in iteritems:
            if fVector_init[eachitem, 0] != 0:
                try:
                    fVector_final = fVector_final + scipy.sparse.csc_matrix(json.loads(rds_ram_handler.hget(int(eachitem/block_size), eachitem)))*fVector_init[eachitem, 0]
                except Exception, e:
                    print e
                    pdb.set_trace()
                    sys.exit()

        fVector_final = fVector_final.toarray().tolist()[0]
        fVector_final = dict(zip(range(0, self.itemnum), fVector_final))

        return fVector_final

    def single_predict(self, uID, rds_ram_handler, groupid):
        fVector = self.calc_RVector(uID, rds_ram_handler, groupid)
        # calc auc
        positive_itemid = self.testSet[uID][0]
        if positive_itemid >= self.itemnum:# item only appearing in test dataset
            fVector =  sorted(fVector.iteritems(), key=lambda d:d[1], reverse = True)
            single_rankingscore = -1# invalid
            auc = -1# invalid
        else:
            negative_itemid = random.randint(0, self.itemnum-1)
            while negative_itemid == positive_itemid:
                negative_itemid = random.randint(0, self.itemnum-1)
            if fVector[positive_itemid] > fVector[negative_itemid]:
                auc = 1.0
            elif fVector[positive_itemid] == fVector[negative_itemid]:
                auc = 0.5
            else:
                auc = 0.0

            fVector =  sorted(fVector.iteritems(), key=lambda d:d[1], reverse = True)
            try:
                single_rankingscore = (fVector.index((positive_itemid, dict(fVector)[positive_itemid])) + 1)/float(len(fVector))
            except Exception, e:
                single_rankingscore = -1# invalid
        return [[each[0] for each in fVector[:10]], single_rankingscore, auc]


    def predict(self, scope, groupid):
        try:
            if self.dataset_name == "caixin":
                rds_ram = redis.client.StrictRedis(host="localhost", port=6379, db=0)
                # rds_rop = redis.client.StrictRedis(host="localhost", port=6379, db=1)# result of predict
            elif self.dataset_name == "fengniao":
                rds_ram = redis.client.StrictRedis(host="localhost", port=6380, db=0)
                # rds_rop = redis.client.StrictRedis(host="localhost", port=6380, db=1)# result of predict
            elif self.dataset_name == "xici":
                rds_ram = redis.client.StrictRedis(host="localhost", port=6381, db=0)
                # rds_rop = redis.client.StrictRedis(host="localhost", port=6381, db=1)# result of predict
            else:
                print "dataset name arg error !"
                sys.exit()
        except Exception, e:
            print e
            sys.exit()

        user_rangkingscore = {}
        user_auc = {}
        user_predictitem = {}
        # pdb.set_trace()
        for user in range(scope[0], scope[1]):
            [predict_item, single_rankingscore, single_auc] = self.single_predict(int(user), rds_ram, groupid)
            user_rangkingscore[user] = single_rankingscore
            user_auc[user] = single_auc
            user_predictitem[user] = predict_item

        # try:
        #     rds_rop.hmset(groupid, user_predictitem)
        # except Exception, e:
        #     print e
        #     sys.exit()
        return [user_predictitem, user_rangkingscore, user_auc]

    def calc_precision(self, scope, groupid, l):
        filepath = "../offline_results/%s/"%self.dataset_name
        try:
            assert type(l) == type([])
        except Exception, e:
            print "l arg error !"
            sys.exit()

        try:
            recommend_list = json.loads(self.read_data(filepath+"user_predictitem.json"))
        except Exception, e:
            print e
            sys.exit()

        #calc precision
        precision_epl = {}
        for each_l in l:
            precision = 0.0
            validuser_count = 0
            for user in range(scope[0], scope[1]):
                positive_itemid = self.testSet[user][0]
                if positive_itemid >= self.itemnum:
                    pass
                else:
                    validuser_count += 1
                    if positive_itemid in recommend_list[str(user)][:each_l]:
                        precision += self.itemnum/each_l
                    else:
                        pass
            precision_epl[each_l] = precision/validuser_count

        try:
            self.store_data(json.dumps({"precision_epl":precision_epl}), filepath + "precision_epl.json")
        except Exception, e:
            print e
            sys.exit()
        return precision_epl

    def calc_rankingscore(self, scope, groupid):
        filepath = "../offline_results/%s/"%self.dataset_name
        try:
            user_rankingscore = json.loads(self.read_data(filepath+"user_rankingscore.json"))
        except Exception, e:
            print e
            sys.exit()

        #calc rankingscore
        rankingscore = 0.0
        validuser_count = 0
        for user in range(scope[0], scope[1]):
            tmp_rankingscore = user_rankingscore[str(user)]
            if tmp_rankingscore == -1:
                pass
            else:
                validuser_count += 1
                rankingscore += tmp_rankingscore
        rankingscore /= validuser_count

        try:
            self.store_data(json.dumps({"rankingscore":rankingscore}), filepath + "rankingscore.json")
        except Exception, e:
            print e
            sys.exit()
        return rankingscore

    def calc_auc(self, scope, groupid, num_randomsample):
        filepath = "../offline_results/%s/"%self.dataset_name
        try:
            assert type(num_randomsample) == type(0)
        except Exception, e:
            print "num_randomsample arg error !"
            sys.exit()


        try:
            user_auc = json.loads(self.read_data(filepath+"user_auc.json"))
        except Exception, e:
            print e
            sys.exit()

        # calc auc
        auc = 0.0
        validuser_count = 0
        # for user in range(scope[0], scope[1]):
        while validuser_count < num_randomsample:
            tmp_auc = user_auc[str(random.randint(scope[0], scope[1]-1))]
            if tmp_auc == -1:
                pass
            else:
                validuser_count += 1
                auc += tmp_auc
        auc /= validuser_count

        try:
            self.store_data(json.dumps({"auc":auc}), filepath + "auc.json")
        except Exception, e:
            print e
            sys.exit()
        return auc

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
    md = MassDiffusion(filepath="../../../../data/daduchouyangshuju1/xicichouyang/", dataset_name="xici", split_traintest="no")
    t0 = time.clock()
    md.import_data(method="online")
    t1 = time.clock()
    print "import_data costs: %ss"%(t1-t0)

    t0 = time.clock()
    md.create_ui_matrix(method="online")
    t1 = time.clock()
    print "create_ui_matrix costs: %ss"%(t1-t0)
 
    # t0 = time.clock()
    # md.calc_RAMatrix((0, 2), 0)
    # t1 = time.clock()
    # print "calc_RAMatrix costs: %ss"%(t1-t0)

    t0 = time.clock()
    md.predict((0, 3), 0)
    t1 = time.clock()
    print "predict costs: %ss"%(t1-t0)
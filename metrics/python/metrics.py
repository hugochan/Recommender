#!/usr/bin/env python
#encoding=utf-8

import sys
import json
import random
import pp
import pdb


class metrics(object):
    """docstring for metrics"""
    def __init__(self, filepath, dataset_name, algorithm_name, decay_factor=0):
        super(metrics, self).__init__()
        self._filepath = filepath
        self.dataset_name = dataset_name
        self.algorithm_name = algorithm_name
        self.user_predictitem = {}
        self.user_rankingscore = {}
        self.user_auc = {}
        if self.algorithm_name == "tra_hc":
            self.decay_factor = decay_factor

    def import_probeset(self):
        try:
            self.probeset = self.read_data(self._filepath + "probeset.json")
            statistics = self.read_data(self._filepath + "statistics.json")
        except Exception, e:
            print e
            sys.exit()

        if self.probeset != "" and statistics != "":
            try:
                self.probeset = json.loads(self.probeset)
                statistics = json.loads(statistics)
            except Exception, e:
                print e
                sys.exit()
        else:
            print "read nothing!"
            sys.exit()

        self.usernum = statistics["usernum"]
        self.itemnum = statistics["train itemnum"]
        self.instancenum = statistics["train instancenum"]
        # self.itemnum = statistics["itemnum"]
        # self.instancenum = statistics["instancenum"]


    def calc(self, triple, position, filename, sampling_times):
        recommendscore = {}
        user_predictitem = {}
        user_rankingscore = {}
        user_auc = {}
        user = triple[0]
        recommendscore[triple[1]] = triple[2]
        try:
            with open(self._filepath+filename, "r") as f:
                f.seek(position)
                iid = 1
                while iid < self.itemnum:
                    for each in range(self.usernum):
                        templine = f.readline()
                    temp = templine[:-1].split("    ")[:3]
                    if int(temp[0]) != user:
                        print "format error !"
                        sys.exit()
                    else:
                        item = int(temp[1])
                        score = float(temp[2])
                        recommendscore[item] = score
        except Exception, e:
            print e
            sys.exit()

        # handle
        positive_itemid = self.probeset[str(user)][0]
        if positive_itemid >= self.itemnum:# item only appearing in probe dataset
            recommendscore =  sorted(recommendscore.iteritems(), key=lambda d:d[1], reverse = True)
            single_rankingscore = -1# invalid
            auc = -1# invalid
        else:
            auc = 0
            for each in range(sampling_times):
                negative_itemid = random.randint(0, self.itemnum-1)
                while negative_itemid == positive_itemid:
                    negative_itemid = random.randint(0, self.itemnum-1)
                if recommendscore[positive_itemid] > recommendscore[negative_itemid]:
                    auc += 1.0
                elif recommendscore[positive_itemid] == recommendscore[negative_itemid]:
                    auc += 0.5
                else:
                    auc += 0.0
            auc /= sampling_times
            
            recommendscore =  sorted(recommendscore.iteritems(), key=lambda d:d[1], reverse = True)
            try:
                single_rankingscore = (recommendscore.index((positive_itemid, dict(recommendscore)[positive_itemid])) + 1)/float(len(recommendscore))
            except Exception, e:
                single_rankingscore = -1# invalid
        
        user_predictitem[user] = [each[0] for each in recommendscore[:20]]
        user_rankingscore[user] = single_rankingscore
        user_auc[user] = auc
        try:
            with open(self._filepath+"user_%s.txt"%user, "w") as f:
                pass
        except:
            print "user:%s error !"%user
            sys.exit()
        f.close()
        return [user_predictitem, user_rankingscore, user_auc]

    def para_statistics(self, result):
        self.user_predictitem.update(result[0])
        self.user_rankingscore.update(result[1])
        self.user_auc.update(result[2])


    def import_recommendscore(self, filename, format, sampling_times):
        user_rankingscore = {}
        user_auc = {}
        user_predictitem = {}

        try:
            with open(self._filepath+filename, "r") as f:
                if format == "user_based":
                    templine = f.readline()
                    while (templine):
                        # read user by user
                        recommendscore = {}
                        temp = templine[:-1].split("    ")[:3]
                        user = int(temp[0])
                        item = int(temp[1])
                        score = float(temp[2])
                        recommendscore[item] = score
                        
                        iid = 1
                        while iid < self.itemnum:
                            templine = f.readline()
                            temp = templine[:-1].split("    ")[:3]
                            if int(temp[0]) != user:
                                print "recommendscore file format error !"
                                sys.exit()
                            item = int(temp[1])
                            score = float(temp[2])
                            recommendscore[item] = score
                            iid += 1

                        # handle
                        positive_itemid = self.probeset[str(user)][0]
                        if positive_itemid >= self.itemnum:# item only appearing in probe dataset
                            recommendscore =  sorted(recommendscore.iteritems(), key=lambda d:d[1], reverse = True)
                            single_rankingscore = -1# invalid
                            auc = -1# invalid
                        else:
                            auc = 0
                            for each in range(sampling_times):
                                negative_itemid = random.randint(0, self.itemnum-1)
                                while negative_itemid == positive_itemid:
                                    negative_itemid = random.randint(0, self.itemnum-1)
                                if recommendscore[positive_itemid] > recommendscore[negative_itemid]:
                                    auc += 1.0
                                elif recommendscore[positive_itemid] == recommendscore[negative_itemid]:
                                    auc += 0.5
                                else:
                                    auc += 0.0
                            auc /= sampling_times

                            recommendscore =  sorted(recommendscore.iteritems(), key=lambda d:d[1], reverse = True)
                            try:
                                single_rankingscore = (recommendscore.index((positive_itemid, dict(recommendscore)[positive_itemid])) + 1)/float(len(recommendscore))
                            except Exception, e:
                                single_rankingscore = -1# invalid

                        user_predictitem[user] = [each[0] for each in recommendscore[:20]]
                        user_rankingscore[user] = single_rankingscore
                        user_auc[user] = auc

                        templine = f.readline()
                
                elif format == "item_based":
                    pdb.set_trace()
                    user_list = []
                    # job_server = pp.Server()# require parallel python
                    templine = f.readline()
                    while (templine):
                        temp = templine[:-1].split("    ")[:3]
                        user = int(temp[0])
                        item = int(temp[1])
                        score = float(temp[2])
                        if user not in user_list:
                            job = job_server.submit(func=self.calc, \
                                args=((user, item, score), f.tell(), filename, sampling_times), \
                                    depfuncs=(), modules=("random",), \
                                        callback=self.para_statistics)
                            user_list.append(user)
                        if len(user_list) == self.usernum:
                            break
                        templine = f.readline()
                    job_server.wait()
                else:
                    print "format arg error !"
                    sys.exit()
        except Exception, e:
            print e
            sys.exit()
        f.close()

        try:
            if format == "user_based":
                self.store_data(json.dumps(user_predictitem), self._filepath + "user_predictitem.json")
                self.store_data(json.dumps(user_rankingscore), self._filepath + "user_rankingscore.json")
                self.store_data(json.dumps(user_auc), self._filepath + "user_auc.json")
            elif format == "time_based":
                self.store_data(json.dumps(self.user_predictitem), self._filepath + "user_predictitem.json")
                self.store_data(json.dumps(self.user_rankingscore), self._filepath + "user_rankingscore.json")
                self.store_data(json.dumps(self.user_auc), self._filepath + "user_auc.json")
            else:
                print "format arg error !"
                sys.exit()
        except Exception, e:
            print e
            sys.exit()

    def calc_auc(self, scope, num_randomsample):
        try:
            assert type(num_randomsample) == type(0)
        except Exception, e:
            print "num_randomsample arg error !"
            sys.exit()


        try:
            user_auc = json.loads(self.read_data(self._filepath+"user_auc.json"))
        except Exception, e:
            print e
            sys.exit()

        # calc auc
        auc = 0.0
        validuser_count = 0
        # while validuser_count < num_randomsample:
        #     tmp_auc = user_auc[str(random.randint(scope[0], scope[1]-1))]
        #     if tmp_auc == -1:
        #         pass
        #     else:
        #         validuser_count += 1
        #         auc += tmp_auc
        # auc /= validuser_count
        for eachuser in range(self.usernum):
            tmp_auc = user_auc[str(eachuser)]
            if tmp_auc == -1:
                pass
            else:
                validuser_count += 1
                auc += tmp_auc
        auc /= validuser_count    
        try:
            self.store_data(json.dumps({"auc":auc}), self._filepath + "auc.json")
        except Exception, e:
            print e
            sys.exit()
        return auc

    def calc_precision_epl(self, scope, l):
        try:
            assert type(l) == type([])
        except Exception, e:
            print "l arg error !"
            sys.exit()

        try:
            recommend_list = json.loads(self.read_data(self._filepath+"user_predictitem.json"))
        except Exception, e:
            print e
            sys.exit()

        #calc precision
        precision_epl = {}
        for each_l in l:
            precision = 0.0
            validuser_count = 0
            for user in range(scope[0], scope[1]):
                positive_itemid = self.probeset[str(user)][0]
                if positive_itemid >= self.itemnum:
                    pass
                else:
                    validuser_count += 1
                    if positive_itemid in recommend_list[str(user)][:each_l]:
                        precision += self.itemnum/float(each_l)
                    else:
                        pass
            precision_epl[each_l] = precision/validuser_count

        try:
            self.store_data(json.dumps({"precision_epl":precision_epl}), self._filepath + "precision_epl.json")
        except Exception, e:
            print e
            sys.exit()
        return precision_epl

    def calc_rankingscore(self, scope):
        try:
            user_rankingscore = json.loads(self.read_data(self._filepath+"user_rankingscore.json"))
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
            self.store_data(json.dumps({"rankingscore":rankingscore}), self._filepath + "rankingscore.json")
        except Exception, e:
            print e
            sys.exit()
        return rankingscore

    def test(self, display=True):
        auc = self.calc_auc((0, self.usernum), self.usernum)
        precision_epl = self.calc_precision_epl((0, self.usernum), range(1, 21))
        rankingscore = self.calc_rankingscore((0, self.usernum))
        if display == True:
            print "dataset: %s"%self.dataset_name
            print "user num: %s"%self.usernum
            print "item num: %s"%self.itemnum
            print "instance num: %s"%self.instancenum
            print "algorithm: %s"%self.algorithm_name
            if self.algorithm_name == "tra_hc":
                print "decay factor: %s"%self.decay_factor
            print "auc: %s"%auc
            for k, v in precision_epl.iteritems():
                print "precision_epl: %s    l: %s"%(v, k)
            print "rankingscore: %s"%rankingscore


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
    decay_factor = -0.5
    import time
    t0=time.clock()
    mymetric = metrics(filepath="../../HeatConduction/offline_results/caixin/hc/", dataset_name="caixin", algorithm_name="hc", decay_factor=decay_factor)
    mymetric.import_probeset()
    # mymetric.import_recommendscore("tra_ucf-user_recommendscore_%s.txt"%decay_factor)
    mymetric.import_recommendscore("hc-item_recommendscore_0.txt", "item_based", 50)
    mymetric.test()
    print "metrics costs %ss"%(time.clock()-t0)

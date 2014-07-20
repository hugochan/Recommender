#!/usr/bin/env python
#encoding=utf-8

import sys, time, json
import scipy.sparse
import numpy
import random
import pdb

class sampling(object):
    """sampling"""
    def __init__(self, filepath, dataset_name, split_trainprobe):
        super(sampling, self).__init__()
        self._filepath = filepath
        self.dataset_name = dataset_name
        self.split_trainprobe = split_trainprobe
        self.userset = {}# user set: {"user":index,...}
        self.itemset = {}# item set: {"item":index,...}
        self.trainset = {}# instance data set: {user:[item,...],...}
        self.probeset = {}

    def import_datas(self):
        try:
            with open(self._filepath+"caixingwang_filtering_0606.txt", 'r') as f:
                tmp_itemset = []# item set
                self.instanceset = {}
                origin_instancenum = 0
                templine = f.readline()
                while(templine):
                    origin_instancenum += 1
                    # temp = templine.split('\t')[:4]
                    temp = templine.split('\t')[:3]

                    user = int(temp[0])
                    item = int(temp[1])
                    # time_stamp = int(temp[3])
                    time_stamp = int(temp[2][:-1])
                    tmp_itemset.append(item)
                    # if int(temp[2]) >= 3:
                    #     try:
                    #         self.instanceset[user].append([item, time_stamp])
                    #     except:
                    #         self.instanceset[user] = [[item, time_stamp]]
                    try:
                        self.instanceset[user].append([item, time_stamp])
                    except:
                        self.instanceset[user] = [[item, time_stamp]]
                    templine = f.readline()
        except Exception, e:
            print "import datas error !"
            print e
            sys.exit()
        f.close()
        self.origin_userset = self.instanceset.keys()
        self.origin_usernum = len(self.origin_userset)
        origin_itemnum = len(list(set(tmp_itemset)))# remove redundancy

        print "origin datas:"
        print "user num: %s"%self.origin_usernum
        print "item num: %s"%origin_itemnum
        print "instance num: %s"%origin_instancenum

    def sample(self, sampling_ratio):
        # sampling
        # 1 navie sampling
        self.usernum = int(self.origin_usernum*sampling_ratio)
        sampling_userset = random.sample(self.origin_userset, self.usernum)
        for each in self.origin_userset:
            if each not in sampling_userset:
                del self.instanceset[each]

        tmp_itemset = []
        for k, v in self.instanceset.iteritems():
            for each in v:
                tmp_itemset.append(each[0])
        self.itemnum = len(set(tmp_itemset))

        iterator = self.instanceset.iteritems()
        self.instanceset = {}# release the space
        sampling_instancenum = 0
        if self.split_trainprobe == "yes":
            trainset = {}
            probeset = {}
            for user, records in iterator:
                i = 0
                recordnum = len(records)
                for eachrecord in sorted(records, key=lambda d:d[1],reverse = False):
                    if i != recordnum - 1:
                        try:
                            strainset[user].append([eachrecord[0], eachrecord[1]])
                        except:
                            trainset[user] = [[eachrecord[0], eachrecord[1]]]
                    else:
                        probeset[user] = [[eachrecord[0], eachrecord[1]]]
                    i += 1
                    sampling_instancenum += 1

            try:
                with open(self._filepath+"%s_samplingtrain_%s.txt"%(self.dataset_name, sampling_ratio), "w") as f:
                    for user, records in trainset.iteritems():
                        data = []
                        for item, time_stamp in records:
                            data.append("%s    %s    %s\n"%(user, item, time_stamp)) 
                        f.writelines(data)
            except Exception, e:
                print e
                sys.exit()

            try:
                with open(self._filepath+"%s_samplingprobe_%s.txt"%(self.dataset_name, sampling_ratio), "w") as f:
                    for user, records in probeset.iteritems():
                        data = []
                        for item, time_stamp in records:
                            data.append("%s    %s    %s\n"%(user, item, time_stamp)) 
                        f.writelines(data)
            except Exception, e:
                print e
                sys.exit()

        
        elif self.split_trainprobe == "no":
            sampling_instanceset = {}
            for user, records in iterator:
                for eachrecord in sorted(records, key=lambda d:d[1],reverse = False):
                    try:
                        sampling_instanceset[user].append([eachrecord[0], eachrecord[1]])
                    except:
                        sampling_instanceset[user] = [[eachrecord[0], eachrecord[1]]]
                    sampling_instancenum  += 1

            try:
                with open(self._filepath+"%s_sampling_%s.txt"%(self.dataset_name, sampling_ratio), "w") as f:
                    for user, records in sampling_instanceset.iteritems():
                        data = []
                        for item, time_stamp in records:
                            data.append("%s    %s    %s\n"%(user, item, time_stamp)) 
                        f.writelines(data)
            except Exception, e:
                print e
                sys.exit()

        else:
            print "split_trainprobe arg error !"
            sys.exit()

        print "after sampling:"
        print "user num: %s"%self.usernum
        print "item num: %s"%self.itemnum
        print "instance num: %s"%sampling_instancenum

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
    sampling = sampling(filepath="../../../data/caixin/", dataset_name="caixin", split_trainprobe="no")
    sampling.import_datas()
    sampling.sample(sampling_ratio=0.2)
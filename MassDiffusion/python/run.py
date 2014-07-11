#!/usr/bin/env python
#encoding=utf-8

import sys, time, json
import threading, pp
from massdiffusion import *
import pdb


class RunRecommender(object):
    """docstring for RunRecommender"""
    def __init__(self, recommender_name="", arg=()):
        super(RunRecommender, self).__init__()
        self.recommender_name = recommender_name
        self.user_predictitem = {}
        self.user_rankingscore = {}
        self.user_auc = {}
        if self.recommender_name == "mass diffusion":
            self.recommender = MassDiffusion(filepath=arg[0], dataset_name=arg[1], split_traintest=arg[2])
            print "dataset name:%s"%arg[1]
            # import datas
            t0 = time.clock()
            self.recommender.import_data(method="online")
            t1 = time.clock()
            print "import_data costs: %ss"%(t1-t0)

            # create ui_matrix
            t0 = time.clock()
            self.recommender.create_ui_matrix(method="online")
            t1 = time.clock()
            print "create_ui_matrix costs: %ss"%(t1-t0)

        else:
            print "recommender arg error !"
            sys.exit()


    def para_predict(self, result):
        self.user_predictitem.update(result[0])
        self.user_rankingscore.update(result[1])
        self.user_auc.update(result[2])

    def train(self, block_size):
        # calc RAMatrix
        t0 = time.clock()
        # using parallel computing
        task_num = int(self.recommender.itemnum/block_size)
        job_server = pp.Server()# require parallel python

        tasknum_per_batch = 10
        batch_num = int(task_num/tasknum_per_batch)
        for each in range(0, batch_num):
            for eachtask in range(each*tasknum_per_batch, (each+1)*tasknum_per_batch):
                job = job_server.submit(func=self.recommender.calc_RAMatrix, \
                    args=((eachtask*block_size, (eachtask+1)*block_size), eachtask), \
                        depfuncs=(), modules=("redis.client", "scipy.sparse", "scipy.io", "json", "sys"))
            
            job_server.wait()
            print "task:%s done !"%eachtask

        if task_num - batch_num*tasknum_per_batch != 0:
            for eachtask in range(batch_num*tasknum_per_batch, task_num):
                job = job_server.submit(func=self.recommender.calc_RAMatrix, \
                    args=((eachtask*block_size, (eachtask+1)*block_size), eachtask), \
                        depfuncs=(), modules=("redis.client", "scipy.sparse", "scipy.io", "json", "sys"))

            job_server.wait()
            print "task:%s done !"%eachtask

        if self.recommender.itemnum - task_num*block_size != 0:
            job = job_server.submit(func=self.recommender.calc_RAMatrix, \
                args=(((eachtask+1)*block_size, self.recommender.itemnum), eachtask+1), \
                    depfuncs=(), modules=("redis.client", "scipy.sparse", "scipy.io", "json", "sys"))
            job_server.wait()
            print "task:%s done !"%(eachtask+1)

        t1 = time.clock()
        print "calc_RAMatrix costs: %ss"%(t1-t0)
        # pdb.set_trace()

        print "saving database..."
        try:
            if self.recommender.dataset_name == "caixin":
                rds_ram = redis.StrictRedis(host="localhost", port=6379, db=0)#ra matrix
            elif self.recommender.dataset_name == "fengniao":
                rds_ram = redis.StrictRedis(host="localhost", port=6380, db=0)#ra matrix
            elif self.recommender.dataset_name == "xici":
                rds_ram = redis.StrictRedis(host="localhost", port=6381, db=0)#ra matrix
            else:
                print "dataset name arg error !"
                sys.exit()
        except Exception, e:
            print e
            sys.exit()
        rds_ram.save()
        print "saved !"


    def test(self, block_size):
        filepath = "../offline_results/%s/"%self.recommender.dataset_name
        t0 = time.clock()
        # using parallel computing
        task_num = int(self.recommender.usernum/block_size)
        job_server = pp.Server()# require parallel python

        tasknum_per_batch = 10
        batch_num = int(task_num/tasknum_per_batch)
        for each in range(0, batch_num):
            for eachtask in range(each*tasknum_per_batch, (each+1)*tasknum_per_batch):
                job = job_server.submit(func=self.recommender.predict, \
                    args=((eachtask*block_size, (eachtask+1)*block_size), eachtask), \
                        depfuncs=(), modules=("redis.client", "scipy.sparse", "scipy.io", "json", "sys", "random"), \
                            callback=self.para_predict)

            job_server.wait()
            print "task:%s done !"%eachtask

        if task_num - batch_num*tasknum_per_batch != 0:
            for eachtask in range(batch_num*tasknum_per_batch, task_num):
                job = job_server.submit(func=self.recommender.predict, \
                    args=((eachtask*block_size, (eachtask+1)*block_size), eachtask), \
                        depfuncs=(), modules=("redis.client", "scipy.sparse", "scipy.io", "json", "sys", "random"), \
                            callback=self.para_predict)
 
            job_server.wait()
            print "task:%s done !"%eachtask

        if self.recommender.usernum - task_num*block_size != 0:
            job = job_server.submit(func=self.recommender.predict, \
                args=(((eachtask+1)*block_size, self.recommender.usernum), eachtask+1), \
                    depfuncs=(), modules=("redis.client", "scipy.sparse", "scipy.io", "json", "sys", "random"), \
                        callback=self.para_predict)

            job_server.wait()
            print "task:%s done !"%(eachtask+1)

        t1 = time.clock()
        print "para_predict costs: %s"%(t1-t0)

        print "saving results..."
        try:
            self.recommender.store_data(json.dumps(self.user_predictitem), filepath + "user_predictitem.json")
            self.recommender.store_data(json.dumps(self.user_rankingscore), filepath + "user_rankingscore.json")
            self.recommender.store_data(json.dumps(self.user_auc), filepath + "user_auc.json")
        except Exception, e:
            print e
            sys.exit()
        print "saved !"

        # print "saving database..."
        # try:
        #     if self.recommender.dataset_name == "caixin":
        #         rds_rop = redis.StrictRedis(host="localhost", port=6379, db=1)# results of predict
        #     elif self.recommender.dataset_name == "fengniao":
        #         rds_rop = redis.StrictRedis(host="localhost", port=6380, db=1)
        #     elif self.recommender.dataset_name == "xici":
        #         rds_rop = redis.StrictRedis(host="localhost", port=6381, db=1)
        #     else:
        #         print "dataset name arg error !"
        #         sys.exit()
        # except Exception, e:
        #     print e
        #     sys.exit()
        # rds_rop.save()
        # print "saved !"

    def calc_precision(self, l):
        t0 = time.clock()
        precision = self.recommender.calc_precision(scope=(0, self.recommender.usernum), groupid=0, l=10)
        t1 = time.clock()
        print "precision_epl: %s"%precision
        print "calc_precision costs: %ss"%(t1-t0)

if __name__ == '__main__':
    myrun = RunRecommender(recommender_name="mass diffusion", arg=(\
            "../../../../data/daduchouyangshuju1/xicichouyang/", \
            "xici", "no"))
    # myrun.train(100)
    myrun.test(40)
    # myrun.calc_precision(l=10)
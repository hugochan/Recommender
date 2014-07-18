#!/usr/bin/env python
#encoding=utf-8

import sys, time, json, os, shutil
import threading, pp
from ucf import *
from tsn_ucf import *
from tra_ucf import *
import pdb


class RunRecommender(object):
    """docstring for RunRecommender"""
    def __init__(self, recommender_name="", arg=()):
        super(RunRecommender, self).__init__()
        self.recommender_name = recommender_name
        if self.recommender_name == "ucf":
            self.recommender = ucf(filepath=arg[0], dataset_name=arg[1], split_trainprobe=arg[2])
            print "dataset name:%s"%arg[1]
            # import datas
            t0 = time.clock()
            self.recommender.import_datas(method="online")
            t1 = time.clock()
            print "import_data costs: %ss"%(t1-t0)

            # create ui_matrix
            t0 = time.clock()
            self.recommender.create_ui_matrix(method="online")
            t1 = time.clock()
            print "create_uit_matrix costs: %ss"%(t1-t0)

        elif self.recommender_name == "tsn_ucf":
            self.recommender = tsn_ucf(filepath=arg[0], dataset_name=arg[1], split_trainprobe=arg[2])
            print "dataset name:%s"%arg[1]
            # import datas
            t0 = time.clock()
            self.recommender.import_datas(method="online")
            t1 = time.clock()
            print "import_data costs: %ss"%(t1-t0)

            # create ui_matrix
            t0 = time.clock()
            self.recommender.create_uit_matrix(method="online")
            t1 = time.clock()
            print "create_uit_matrix costs: %ss"%(t1-t0)


        elif self.recommender_name == "tra_ucf":
            self.recommender = tra_ucf(filepath=arg[0], dataset_name=arg[1], split_trainprobe=arg[2])
            print "dataset name:%s"%arg[1]
            # import datas
            t0 = time.clock()
            self.recommender.import_datas(method="online")
            t1 = time.clock()
            print "import_data costs: %ss"%(t1-t0)

            # create ui_matrix
            t0 = time.clock()
            self.recommender.create_uit_matrix(method="online")
            t1 = time.clock()
            print "create_uit_matrix costs: %ss"%(t1-t0)

        else:
            print "recommender arg error !"
            sys.exit()


    def para_predict(self, result):
        pass

    def run_recommend(self, block_size):
        t0 = time.clock()
        # temp file
        filepath = "../offline_results/%s/%s/"%(self.recommender.dataset_name, self.recommender_name)
        if not os.path.exists(filepath+"temp/"):
            os.makedirs(filepath+"temp/")

        # using parallel computing
        task_num = int(self.recommender.usernum/block_size)
        job_server = pp.Server()# require parallel python

        tasknum_per_batch = 20
        batch_num = int(task_num/tasknum_per_batch)
        for each in range(0, batch_num):
            for eachtask in range(each*tasknum_per_batch, (each+1)*tasknum_per_batch):
                job = job_server.submit(func=self.recommender.recommend, \
                    args=((eachtask*block_size, (eachtask+1)*block_size), eachtask), \
                        depfuncs=(), modules=("numpy", "scipy.sparse", "sys"), \
                            callback=self.para_predict)

            job_server.wait()
            print "task:%s done !"%eachtask

        if task_num - batch_num*tasknum_per_batch != 0:
            for eachtask in range(batch_num*tasknum_per_batch, task_num):
                job = job_server.submit(func=self.recommender.recommend, \
                    args=((eachtask*block_size, (eachtask+1)*block_size), eachtask), \
                        depfuncs=(), modules=("numpy", "scipy.sparse", "sys"), \
                            callback=self.para_predict)
 
            job_server.wait()
            print "task:%s done !"%eachtask

        if self.recommender.usernum - task_num*block_size != 0:
            eachtask += 1
            job = job_server.submit(func=self.recommender.recommend, \
                args=((eachtask*block_size, self.recommender.usernum), eachtask), \
                    depfuncs=(), modules=("numpy", "scipy.sparse", "sys"), \
                        callback=self.para_predict)

            job_server.wait()
            print "task:%s done !"%eachtask

        # create recommendscore file
        try:
            fhandler = open(filepath+"%s-user_recommendscore.txt"%self.recommender_name, "w")
        except Exception, e:
            print e
            sys.exit()

        for fid in range(eachtask+1):
            try:
                with open(filepath+"temp/"+"%s-user_recommendscore.txt_%s"%(self.recommender_name, fid), "r") as f:
                    data = f.read()
                    if data != "":
                        fhandler.write(data)
                        data = ""
            except Exception, e:
                print e
                sys.exit()
            f.close()
        fhandler.close()

        #delete temp file
        shutil.rmtree(filepath+"temp/")

        t1 = time.clock()
        print "run_recommend costs: %s"%(t1-t0)


if __name__ == '__main__':
    myrun = RunRecommender(recommender_name="tra_ucf", arg=(\
            "../../../../../data/public_datas/movielens-100k/ml-100k/", \
            "ml-100k", "yes"))
    myrun.run_recommend(20)
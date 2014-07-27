#!/usr/bin/env python
#encoding=utf-8

import sys, time, json, os, shutil
import threading, pp
from heatconduction import *
from tra_heatconduction import *
import pdb


class RunRecommender(object):
    """docstring for RunRecommender"""
    def __init__(self, recommender_name, arg, decay_factor=0):
        super(RunRecommender, self).__init__()
        self.recommender_name = recommender_name
        self.decay_factor = decay_factor
        if self.recommender_name == "hc":
            self.recommender = HeatConduction(filepath=arg[0], dataset_name=arg[1], split_trainprobe=arg[2])
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

        elif self.recommender_name == "tra_hc":
            self.recommender = TraHeatConduction(filepath=arg[0], dataset_name=arg[1], split_trainprobe=arg[2], decay_factor=decay_factor)
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
        task_num = int(self.recommender.itemnum/block_size)
        job_server = pp.Server()# require parallel python

        tasknum_per_batch = 30
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

        if self.recommender.itemnum - task_num*block_size != 0:
            eachtask += 1
            job = job_server.submit(func=self.recommender.recommend, \
                args=((eachtask*block_size, self.recommender.itemnum), eachtask), \
                    depfuncs=(), modules=("numpy", "scipy.sparse", "sys"), \
                        callback=self.para_predict)

            job_server.wait()
            print "task:%s done !"%eachtask

        # create recommendscore file
        try:
            fhandler = open(filepath+"%s-item_recommendscore_%s.txt"%(self.recommender_name, self.decay_factor), "w")
        except Exception, e:
            print e
            sys.exit()

        for fid in range(eachtask+1):
            try:
                with open(filepath+"temp/"+"%s-item_recommendscore.txt_%s"%(self.recommender_name, fid), "r") as f:
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
    # for each in range(1, 2):
    #     if each == 0:
    #         continue
    #     myrun = RunRecommender(recommender_name="tsn_ucf", arg=(\
    #             "../../../../../data/caixin/", \
    #             "caixin", "yes"), decay_factor=-0.5)
    #     myrun.run_recommend(20)
    # myrun = RunRecommender(recommender_name="hc", arg=(\
    #     "../../../../data/daduchouyangshuju1/xicichouyang/",\
    #         "xici", "no"))
    myrun = RunRecommender(recommender_name="tra_hc", arg=(\
        "../../../../data/daduchouyangshuju1/xicichouyang/",\
            "xici", "no"), decay_factor=-0.4)
    myrun.run_recommend(200)
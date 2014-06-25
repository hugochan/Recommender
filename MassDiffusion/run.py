#!/usr/bin/env python
#encoding=utf-8

import sys, time
import threading, pp
from massdiffusion import *
import pdb


class RunningThread(threading.Thread):
    def __init__(self, target, name, args):  
        super(RunningThread, self).__init__(target=target, name=name, args=args)
        # self.mylock = threading.RLock()
        self.thread_stop = False  

    def stop(self):  
        self.thread_stop = True

class RunRecommender(object):
    """docstring for RunRecommender"""
    def __init__(self, recommender_name="", arg=()):
        super(RunRecommender, self).__init__()
        self.recommender_name = recommender_name
        if self.recommender_name == "mass diffusion":
            self.recommender = MassDiffusion(filepath=arg[0], dataset_name=arg[1], split_traintest=arg[2])

        else:
            print "recommender arg error !"
            sys.exit()


    def para_calc_RAMatrix(self, job):
        job()
        self.threadcnt -= 1

    def train(self):
        # import datas
        t0 = time.clock()
        self.recommender.import_data(method="offline")
        t1 = time.clock()
        print "import_data costs: %ss"%(t1-t0)

        # create ui_matrix
        t0 = time.clock()
        self.recommender.create_ui_matrix(method="offline")
        t1 = time.clock()
        print "create_ui_matrix costs: %ss"%(t1-t0)

        # calc RAMatrix
        t0 = time.clock()
        # using parallel computing
        block_size = 500
        task_num = int(self.recommender.itemnum/block_size)
        job_server = pp.Server()# require parallel python

        # pdb.set_trace()
        tasknum_per_batch = 3
        batch_num = int(task_num/tasknum_per_batch)
        for each in range(batch_num):
            threadList = []
            threadcnt = 0
            for eachtask in range(each*tasknum_per_batch, (each+1)*tasknum_per_batch):
                job = job_server.submit(func=self.recommender.calc_RAMatrix, args=((eachtask*block_size, (eachtask+1)*block_size), eachtask), depfuncs=(), modules=("redis.client", "scipy.sparse", "scipy.io", "json", "sys"))
                time.sleep(2)
                threadList.append(RunningThread(target=self.para_calc_RAMatrix,\
                    name="para_calc_RAMatrix:task_%s"%eachtask,\
                    args=(job,)))

            
            self.threadcnt = len(threadList)
            for eachthread in threadList:
                eachthread.start()

            while self.threadcnt:
                for eachthread in threadList:
                    eachthread.join(0)
                time.sleep(10)

            print "task:"%eachtask

        if task_num - batch_num*tasknum_per_batch != 0:
            threadList = []
            threadcnt = 0
            for eachtask in range(batch_num*tasknum_per_batch, task_num):
                job = job_server.submit(func=self.recommender.calc_RAMatrix, args=((eachtask*block_size, (eachtask+1)*block_size), eachtask), depfuncs=(), modules=("redis.client", "scipy.sparse", "scipy.io", "json", "sys"))
                time.sleep(2)
                threadList.append(RunningThread(target=self.para_calc_RAMatrix,\
                    name="para_calc_RAMatrix:task_%s"%eachtask,\
                    args=(job,)))

            self.threadcnt = len(threadList)
            for eachthread in threadList:
                eachthread.start()

            while self.threadcnt:
                for eachthread in threadList:
                    eachthread.join(0)
                time.sleep(10)
            
            print "task:"%eachtask


        if self.recommender.itemnum - task_num*block_size != 0:
            threadList = []
            threadcnt = 0
            job = job_server.submit(func=self.recommender.calc_RAMatrix, args=(((eachtask+1)*block_size, self.recommender.itemnum), eachtask+1), depfuncs=(), modules=("redis.client", "scipy.sparse", "scipy.io", "json", "sys"))
            time.sleep(2)
            threadList.append(RunningThread(target=self.para_calc_RAMatrix,\
                name="para_calc_RAMatrix:task_%s"%(eachtask+1),\
                args=(job,)))
            
            self.threadcnt = len(threadList)
            for eachthread in threadList:
                eachthread.start()

            while self.threadcnt:
                for eachthread in threadList:
                    eachthread.join(0)
                time.sleep(10)


        t1 = time.clock()
        print "calc_RAMatrix costs: %ss"%(t1-t0)

class MethodProxy(object):
    def __init__(self, obj, method):
        self.obj = obj
        if isinstance(method, basestring):
            self.methodName = method
        else:
            assert callable(method)
            self.methodName = method.func_name

    def __call__(self, *args, **kwargs):
        return getattr(self.obj, self.methodName)(*args, **kwargs)

# picklableMethod = MethodProxy(someObj, someObj.method)

def calc_RAMatrix(scope, handler):
    handler.calc_RAMatrix(scope)

if __name__ == '__main__':
    myrun = RunRecommender(recommender_name="mass diffusion", arg=(\
            "../../../data/shujuchuli_0618/fengniao_chengyu_0618.txt", \
            "fengniao", "yes"))
    myrun.train()
import collections
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
import torch.optim as optim
from numpy import linalg as LA
from tqdm.notebook import tqdm
all_index = None
use_gpu = torch.cuda.is_available()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


def save_result(path, data):
    with open(path, "w") as f:
        f.write(data)
    f.close()
def select_data(pre_matrix, fraction, dst_train, classes, balance=True):
    if balance:
        top_examples = np.array([], dtype=np.int64)
        train_indx = np.arange(len(dst_train))
        top_examples = np.array([], dtype=np.int64)
        for c in range(classes):
            c_indx = train_indx[dst_train.targets==c]
            budget = round(fraction*len(c_indx))
            # top_examples = np.append(top_examples, c_indx[np.argsort()])
    return {"indices":top_examples, "scores":pre_matrix}

# ========================================
#      loading datas
def scaleEachUnitaryDatas(datas):
    norms = datas.norm(dim=2, keepdim=True)
    return datas / norms


def centerDatas(datas, n_lsamples):
    datas[:, :n_lsamples] = datas[:, :n_lsamples, :] - datas[:, :n_lsamples].mean(1, keepdim=True)
    datas[:, :n_lsamples] = datas[:, :n_lsamples, :] / torch.norm(datas[:, :n_lsamples, :], 2, 2)[:, :, None]
    datas[:, n_lsamples:] = datas[:, n_lsamples:, :] - datas[:, n_lsamples:].mean(1, keepdim=True)
    datas[:, n_lsamples:] = datas[:, n_lsamples:, :] / torch.norm(datas[:, n_lsamples:, :], 2, 2)[:, :, None]
    print("=> redifine data label-samples shape : ", datas[:, :n_lsamples, :].shape)
    return datas


def QRreduction(datas):
    
    ndatas = torch.linalg.qr(datas.permute(0,2,1)).R
    ndatas = ndatas.permute(0,2,1)
    return ndatas


class Model:
    def __init__(self, n_ways):
        self.n_ways = n_ways
              
# ---------  GaussianModel
class GaussianModel(Model):
    def __init__(self, n_ways, lam, ndatas, n_runs, n_shot, n_queries, n_nfeat, all_index, labels):
        super(GaussianModel, self).__init__(n_ways)
        self.mus = None         # shape [n_runs][n_ways][n_nfeat]
        self.lam = lam
        self.ndatas = ndatas
        self.n_runs = n_runs
        self.n_shot = n_shot
        self.n_queries = n_queries
        self.n_nfeat = n_nfeat
        self.all_index = all_index
        self.labels = labels
    def clone(self):
        other = GaussianModel(self.n_ways)
        other.mus = self.mus.clone()
        return self

    def cuda(self):
        self.mus = self.mus.cuda()
        
    def initFromLabelledDatas(self):
        self.mus = self.ndatas.reshape(self.n_runs, self.n_shot+self.n_queries,self.n_ways, self.n_nfeat)[:,:self.n_shot,].mean(1)                           

    def updateFromEstimate(self, estimate, alpha):   
        
        Dmus = estimate - self.mus
        self.mus = self.mus + alpha * (Dmus)

    def compute_optimal_transport(self, M, r, c, epsilon=1e-6):
        
        r = r.cuda() # 5 50000
        c = c.cuda() # 5 10
        n_runs, n, m = M.shape  # 5, 50000, 10
        # print("1======> show M :", M.shape)
        # print("2======> show r and c :", r.shape, c.shape)
        P = torch.exp(- self.lam * M)
        P /= P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)

        print("=====> opt description : ", c.shape, r.shape, P.shape)                            
        u = torch.zeros(n_runs, n).cuda()
        maxiters = 1000
        iters = 1
        # normalize this matrix
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:
            u = P.sum(2)
            P *= (r / u).view((n_runs, -1, 1))
            P *= (c / P.sum(1)).view((n_runs, 1, -1))
            if iters == maxiters:
                break
            iters = iters + 1
        return P, torch.sum(P * M)
    
    def getProbas(self):
        # compute squared dist to centroids [n_runs][n_samples][n_ways]
        # dist = (self.ndatas.unsqueeze(2)-self.mus.unsqueeze(1)).norm(dim=3).pow(2)
        print(self.ndatas.shape) #[1,200000,512]
        print(self.mus.shape)    #[1,200,512]
        _, all_samples, features = self.ndatas.shape
        _, num_class, _ = self.mus.shape
        N=10
        dist = torch.zeros((1,all_samples,num_class)).cuda()
        for i in range(N):
            dist[:,i*int(all_samples/N):(i+1)*int(all_samples/N),:] = (self.ndatas[:,i*int(all_samples/N):(i+1)*int(all_samples/N),].unsqueeze(2)-self.mus.unsqueeze(1)).norm(dim=3).pow(2)
        
        
        norm_matrix = torch.zeros([self.n_runs, self.n_ways, self.n_queries], requires_grad=False)
        p_xj = torch.zeros_like(dist)
        n_usamples = self.n_ways * self.n_queries
        n_lsamples = self.n_ways * self.n_shot
        r = torch.ones(self.n_runs, n_usamples)  #
        c = torch.ones(self.n_runs, self.n_ways) * self.n_queries
        print("333=======> ", dist[:,n_lsamples:])
        p_xj_test, _ = self.compute_optimal_transport(dist[:, n_lsamples:], r, c, epsilon=1e-6)
        print("444=======> ", p_xj_test.shape, p_xj_test.norm(dim=2).pow(2).view(self.n_runs,self.n_queries,self.n_ways).shape, p_xj_test.norm(dim=2).pow(2)[0])
        sorted, indices = torch.sort(p_xj_test.norm(dim=2).pow(2)[0])
        cal_norm = p_xj_test.norm(dim=2).pow(2)[0]
        # global all_index
        result = []
        for i in range(self.n_ways):
            # #  using val
            index_save = torch.argsort(cal_norm[self.all_index[i][self.n_shot:]-int(n_lsamples)]).cpu().numpy()
            # print("show__[%d]"%(i),"   ", len(index_save), len(torch.unique(torch.argsort(cal_norm[self.all_index[i][self.n_shot:]-int(n_lsamples)]))), torch.sort(cal_norm[self.all_index[i][self.n_shot:]-int(n_lsamples)]))
            result.append(index_save)
        for i in range(self.n_ways):
            norm_matrix[:, i, :] = p_xj_test[:, i*self.n_queries:(i+1)*self.n_queries, i]
        n_lsamples = self.n_ways * self.n_shot
        p_xj[:, n_lsamples:] = p_xj_test
        #将每一类的5000个样本的正则值存出来，然后进行排序过的索引，在从训练数据中获得索引
        #在将数据保存出来
        p_xj[:,:n_lsamples].fill_(0)
        p_xj[:,:n_lsamples].scatter_(2,self.labels[:,:n_lsamples].unsqueeze(2), 1)
        print("*******************result.shape:", cal_norm.shape)
        return p_xj, cal_norm

    def estimateFromMask(self, mask):

        emus = mask.permute(0,2,1).matmul(self.ndatas).div(mask.sum(dim=1).unsqueeze(2))

        return emus

          
# =========================================
#    MAP
# =========================================

class MAP:
    def __init__(self, labels, n_runs, n_ways, n_shot, n_queries, alpha=None):
        
        self.verbose = False
        self.progressBar = False
        self.alpha = alpha
        self.labels = labels
        self.n_runs = n_runs
        self.n_ways = n_ways
        self.n_shot = n_shot
        self.n_queries = n_queries
    def getAccuracy(self, probas):
        n_lsamples = self.n_ways * self.n_shot
        olabels = probas.argmax(dim=2)
        matches = (self.labels).eq(olabels).float()
        acc_test = matches[:,n_lsamples:].mean(1)    

        m = acc_test.mean().item()
        pm = acc_test.std().item() *1.96 / math.sqrt(self.n_runs)
        return m, pm
    
    def performEpoch(self, model, epochInfo=None):
     
        p_xj, _ = model.getProbas()
        self.probas = p_xj
        
        if self.verbose:
            print("accuracy from filtered probas", self.getAccuracy(self.probas))
        
        m_estimates = model.estimateFromMask(self.probas)
               
        # update centroids
        model.updateFromEstimate(m_estimates, self.alpha)

        if self.verbose:
            op_xj, _ = model.getProbas()
            acc = self.getAccuracy(op_xj)
            print("output model accuracy", acc)
        
    def loop(self, model, n_epochs=20):
        
        self.probas, _ = model.getProbas()
        if self.verbose:
            print("initialisation model accuracy", self.getAccuracy(self.probas))

        if self.progressBar:
            if type(self.progressBar) == bool:
                pb = tqdm(total = n_epochs)
            else:
                pb = self.progressBar
           
        for epoch in range(1, n_epochs+1):
            if self.verbose:
                print("----- epoch[{:3d}]  lr_p: {:0.3f}  lr_m: {:0.3f}".format(epoch, self.alpha))
            self.performEpoch(model, epochInfo=(epoch, n_epochs))
            if (self.progressBar): pb.update()
        
        # get final accuracy and return it
        op_xj, _ = model.getProbas()
        acc = self.getAccuracy(op_xj)
        return acc
    
    


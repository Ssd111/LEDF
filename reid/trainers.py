from __future__ import print_function, absolute_import, division

import copy
import time
import random
from reid.utils.data import transforms as T
import os.path as osp

import numpy as np
import collections
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from reid.loss.triplet import TripletLoss
from .utils.meters import AverageMeter
from .models import *
from .evaluation_metrics import accuracy
from torch.autograd import Variable
from scipy.special import binom as bm


# torch.set_printoptions(profile="full")


# import pylab as pl
# import seaborn as sns
# from sklearn.manifold import TSNE
class Trainer(object):
    def __init__(self, args, model, memory, criterion,criterion2=None,model0=None,model1=None,model2=None,model00=None,model11=None,model22=None,
                 model_ori=None,mmd=None,memory0=None,memory1=None,memory2=None,memory00=None,memory11=None,memory22=None):
        super(Trainer, self).__init__()
        self.model = model
        self.memory = memory
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = criterion
        self.entropy = nn.CrossEntropyLoss()
        self.args = args


    def train_ledf(self, epoch, data_loaders, optimizer, print_freq=10, train_iters=400):
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        source_count = len(data_loaders)
        losses_s0 = AverageMeter()
        losses_tri = AverageMeter()

        end = time.time()

        # ------------------------------------------------------------------------------------------

        for i in range(train_iters):

            # with torch.autograd.set_detect_anomaly(True):
            if True:
                loss_all = torch.tensor([0.])
                data_loader_index = [i for i in range(source_count)]  ## 0 2
                batch_data = [data_loaders[i].next() for i in range(source_count)]
                data_time.update(time.time() - end)
                domain_grads = []
                loss_final = 0.
                grad_new = 0.

                with torch.set_grad_enabled(True):
                    for t in data_loader_index:  # 0 1
                        data_time.update(time.time() - end)
                        traininputs = batch_data[t]
                        inputs1, targets, _, _, dataset_id = self._parse_data(traininputs)
                        f_out, _ = self.model(inputs1, ledf=True, memory=self.memory, t0=t, gt=targets)
                        loss_s = self.memory[t](f_out, targets).mean()
                        loss_mtr_tri1 = self.criterion(f_out, targets)
                        self.model.zero_grad()
                        loss_all = loss_s+loss_mtr_tri1
                        loss_all.backward()
                        grads = self.get_grads()
                        losses_s0.update(loss_s.item())
                        losses_tri.update(loss_mtr_tri1.item())
                        grad_new = grad_new+grads
                grad_new = grad_new/3
                self.model.zero_grad()
                optimizer.zero_grad()
                self.set_grads(grad_new)
                optimizer.step()
                self.model.zero_grad()



                with torch.no_grad():
                    for m_ind in range(source_count):
                        imgs1, pids, _, _, _ = self._parse_data(batch_data[m_ind])
                        f_new,_ = self.model(imgs1)
                        self.memory[m_ind].module.MomentumUpdate(f_new, pids)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss_s0 {:.3f}({:.3f})\t'
                      'Loss_tri {:.3f}({:.3f})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              losses_s0.val, losses_s0.avg,
                              losses_tri.val, losses_tri.avg,
                              ))







    def _parse_data(self, inputs):
        imgs, names, pids, cams, dataset_id, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(), cams.cuda(), dataset_id.cuda()


    def get_grads(self):
        grads = []
        for p in self.model.module.parameters():
            if p.grad is None:
                # print('!!!!!!!!!!')
                continue
            p1 = p.grad.data.clone().flatten()
            grads.append(p1)
        return torch.cat(grads)


    def set_grads(self, new_grads):
        start = 0
        for k, p in enumerate(self.model.module.parameters()):
            if p.grad is None:
                # print("************")
                continue
            dims = p.shape
            end = start + dims.numel()
            p.grad.data = new_grads[start:end].reshape(dims)
            start = end





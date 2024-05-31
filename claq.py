import math
import time

import torch
import torch.nn as nn
import transformers
import yaml
from quant import *


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class CLAQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, bit, log, mixdict, layername, outlier, outlier_col_dynamic, outlier_layer_dynamic, outlierorder, inputhes, save_quant=None, blocksize=128, percdamp=.01, groupsize=-1, actorder=False):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()
        bits = bit if mixdict == 'None' else int(mixdict[layername])
        KM = []
        W_int = []

        '''if not self.quantizer.ready():
            KM_ = torch.zeros(W.size(1),2**bits)
            W_int_ = torch.zeros(W.size(0),W.size(1))
            self.quantizer.find_params(W, KM_.t(), W_int_, weight=True)'''

        H = self.H
        Hinp = H
        H_avgtrace = torch.trace(H)/H.size(0)
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        mean = torch.mean(W.view(-1).float().abs(),dim=0)
        t = 13
        #upper =  W.view(1,-1).sort()[0][0][round(W.numel()*(1-outlier))]
        #lower =  W.view(1,-1).sort()[0][0][round(W.numel()*outlier)]
        #print(upper,lower)
        
        ################################# input hessian ################################
        if inputhes:
            sens =(torch.diag(Hinv).float_power(-2)*W.pow(2)).mean(dim = 0)
            cur =  sens.sort(descending=True)[1]#[0][len(sens)//4]
            if inputhes < 2 or inputhes > 4:
                raise ValueError("mixed-precision bit out of range, try 2~4")
            inputhes_ = (inputhes-2)/2 if inputhes<3 else inputhes-3
            if inputhes < 3: 
                sens[cur[:round(len(sens)*inputhes_)]]=4
                sens[cur[round(len(sens)*inputhes_):]]=2
            else:
                sens[cur[:round(len(sens)*inputhes_)]]=4
                sens[cur[round(len(sens)*inputhes_):]]=3
            out = sens
        ################################# input hessian ################################
        
        if outlier_col_dynamic:
            zero = torch.zeros_like(W) 
            one = torch.ones_like(W)
            sens = torch.sum(torch.where(torch.gt(W, -mean*t) & torch.lt(W, mean*t), zero, one),dim=0)
            #sens =(torch.diag(Hinv).float_power(-2)*W.pow(2)).mean(dim = 0)
            #sens =(Hinp).mean(dim =0)
            par = 0.1
            cur =  sens.sort(descending=True)[1]#[0][len(sens)//4]
            sens[cur[:round(len(sens)*par)]]=0.014
            sens[cur[round(len(sens)*par):round(len(sens)*(1-par))]]=0.004
            sens[cur[round(len(sens)*(1-par)):]]=0.004
            out_per = sens

        ################################# outlier order ################################
        
        if outlierorder:
            zero = torch.zeros_like(W) 
            one = torch.ones_like(W)
            sens = torch.sum(torch.where(torch.gt(W, -mean*t) & torch.lt(W, mean*t), zero, one),dim=0)
            cur =  sens.sort(descending=True)[1]#[0][len(sens)//4]
            if outlierorder < 2 or outlierorder > 4:
                raise ValueError("mixed-precision bit out of range, try 2~4")
            outlierorder_ = (outlierorder-2)/2 if outlierorder<3 else outlierorder-3
            if mixdict:
                sens[cur[:round(len(sens)*outlierorder)]]=3
                sens[cur[round(len(sens)*outlierorder):round(len(sens)*(1-outlierorder))]]=2
                sens[cur[round(len(sens)*(1-outlierorder)):]]=2
            if mixdict =='None':         #simple mode
                if outlierorder < 3: 
                    sens[cur[:round(len(sens)*outlierorder_)]]=4
                    sens[cur[round(len(sens)*outlierorder_):]]=2
                else:
                    sens[cur[:round(len(sens)*outlierorder_)]]=4
                    sens[cur[round(len(sens)*outlierorder_):]]=3  
            out = sens
            
        ################################# Quantization ################################
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            if inputhes or outlierorder:
                out1 = out[i1:i2].clone()
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                if inputhes or outlierorder:
                    bits = int(out1[i])
                if outlier_col_dynamic :
                    outlier = out_per[i].item()
                    if not outlierorder and not inputhes:
                        bits = bit
                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)

                km = kmclustering(w,bits)
                KM.append(km)
                if outlier != 0 and not save_quant:
                    q = quantize_outlier(w.unsqueeze(1), bits, outlier, 1-outlier, mean,km).flatten()

                elif save_quant:
                    q,_ = index_quantize(w.unsqueeze(1), bits, km)
                    q = q.flatten()
                    _ = _.cpu()
                    W_int.append(_)
                else:
                    q = quantize(w.unsqueeze(1), bits, km).flatten()      
            
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))


        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())
        quantlog = log + '/' + log.split('/')[-1] + '_quantinfo.yaml'
        with open(quantlog,'a') as f:
            f.write(yaml.dump({layername:[W.numel(), torch.sum(Losses).item()]}))

        if actorder:
            invperm = torch.argsort(perm)
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if save_quant:
            KM = torch.stack(KM,0)
            W_int = torch.stack(W_int,1)
            km,W_int = KM.cpu(),W_int.cpu()
            self.quantizer.find_params(W_int, KM, weight=True)
            del KM
            del W_int
            torch.cuda.empty_cache()
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()

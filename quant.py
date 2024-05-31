import numpy as np
import torch
import torch.nn as nn
from sklearnex import patch_sklearn, config_context
patch_sklearn()
#from cuml.cluster import KMeans
#import cudf
from sklearn.cluster import KMeans#,MiniBatchKMeans
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def power_quant(x, value_s):  #quantize weight to nearest centroid
    shape = x.shape
    xhard = x.view(-1)
    idxs = (xhard.unsqueeze(0) - value_s.reshape(-1,1)).abs().min(dim=0)[1]
    xhard = value_s[idxs].view(shape)
    return xhard

def index_quant(x, value_s):  #store the index of the codebook
    shape = x.shape
    xhard = x.view(-1)
    idxs = (xhard.unsqueeze(0) - value_s.reshape(-1,1)).abs().min(dim=0)[1]
    return idxs

'''def quantize(x, bit, scale, zero, maxq):  #uniform style
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)'''

def quantize_outlier(x, bit, low, up, mean, q_value): # keep outlier K-Means style fake_quantize
    q = power_quant(x,q_value.to(x.device))
    if low>0 and low<1:
        W1 = x.reshape(1,-1)
        upper =  W1.sort(descending=True)[0][0][round((W1.size(1)*low))]
        lower =  W1.sort(descending=True)[0][0][round((W1.size(1)*up))]
        zero = torch.zeros_like(x) 
        one = torch.ones_like(x)
        outlier = torch.where(torch.gt(x, lower) & torch.lt(x, upper), zero, x)
        musk = torch.where(torch.gt(x, lower) & torch.lt(x, upper), one, zero)
        q = q*musk+outlier

    elif low>1:
        W1 = x.reshape(1,-1)
        cur = torch.mean(x.view(-1).float().abs(),dim=0)*low  #colomn level  
        zero = torch.zeros_like(x) 
        one = torch.ones_like(x)
        outlier = torch.where(torch.gt(x, -cur) & torch.lt(x, cur), zero, x)
        musk = torch.where(torch.gt(x, -cur) & torch.lt(x, cur), one, zero)
        q = q*musk+outlier        
    return q.float()

def kmclustering(x,bit):   #K-Means clustering
    k_means = KMeans(n_clusters=2**bit, random_state=10,n_init='auto')
    k_means.fit(x.reshape(-1,1).cpu())
    q_value = torch.tensor(k_means.cluster_centers_).reshape(-1)
    return q_value

def quantize(x, bit, q_value): # standard K-Means style fake_quantize
    q = power_quant(x,q_value.to(x.device))
    return q.float()

def index_quantize(x, bit, q_value): # standard K-Means style
    q = power_quant(x,q_value.to(x.device))
    center = index_quant(x,q_value.to(x.device))
    return q.float(), center


'''def quantize(x, bit, scale, zero, maxq):#rapid cuml style
    kmeans = KMeans(n_clusters=2**bit, random_state=10,n_init='auto')
    kmeans.fit(x.reshape(-1,1).cpu())
    q_value = torch.tensor(k_means.cluster_centers_).reshape(-1)
    #q_value.sort()
    #print(q_value)
    q = power_quant(x,q_value.to(x.device))
    return q.float().to(device)'''

class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        #self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))
        self.register_buffer('kmvalue', torch.zeros(16))
        self.register_buffer('W_int', torch.zeros(shape))     
        

    def configure(
        self,
        bits, perchannel=False, sym=True, 
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False
    ):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 
        if trits:
            self.maxq = torch.tensor(-1) 

    def find_params(self, W_int, kmvalue, weight=False):
    #def find_params(self, x, kmvalue, weight=False): 
        self.kmvalue = kmvalue
        self.W_int = W_int
        '''dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
          self.scale = xmax
          self.zero = xmin
        else:
          self.scale = (xmax - xmin) / self.maxq
          if self.sym:
              self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
          else:
              self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid 
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1)) 
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)'''

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


try:
    import quant_cuda
except:
    print('CUDA extension not installed.')


########################## Put two 4-bit weight in one 8-bit ###########################
def two_compl(x, bits):
    return torch.where(x < 0, 2 ** bits + x, x)

def pack_to_i4(X):
    X_i8 = two_compl(X.to(dtype=torch.int8), 4).to(torch.uint8)
    X_i4 = X_i8[:, 0::2] | (X_i8[:, 1::2] << 4)
    return X_i4


# Assumes layer is perfectly divisible into 1024 * 1024 blocks
class Quant3Linear(nn.Module): 

    def __init__(self, infeatures, outfeatures, faster=False):
        super().__init__()
        self.register_buffer('kmvalues', torch.zeros((outfeatures, 16)))
        #self.register_buffer('zeros', torch.zeros((outfeatures, 1)))
        #self.register_buffer('scales', torch.zeros((outfeatures, 1)))
        self.register_buffer('bias', torch.zeros(outfeatures))
        #self.register_buffer('qweight', torch.zeros((infeatures // 32 *3, outfeatures), dtype=torch.int))
        self.register_buffer('qweight', torch.randint(1, 7, (outfeatures, infeatures // 2),
            dtype=torch.uint8, requires_grad=False))
        self.faster = faster

    def pack(self, linear, kmvalues, W_int):
        self.kmvalues = kmvalues
        self.W_int = W_int
        #self.zeros = zeros * scales
        #self.scales = scales.clone()
        if linear.bias is not None:
            self.bias = linear.bias.clone()

        '''
        intweight = torch.round((linear.weight.data + self.zeros) / self.scales).to(torch.int)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros(
            (intweight.shape[0] // 32 * 3, intweight.shape[1]), dtype=np.uint32
        )
        i = 0
        row = 0
        while row < qweight.shape[0]:
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i))
            i += 10
            qweight[row] |= intweight[i] << 30
            row += 1
            qweight[row] |= (intweight[i] >> 2) & 1
            i += 1
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i) + 1)
            i += 10
            qweight[row] |= intweight[i] << 31
            row += 1
            qweight[row] |= (intweight[i] >> 1) & 0x3
            i += 1
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i) + 2)
            i += 10
            row += 1

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight) '''
        self.qweight = pack_to_i4(W_int)

    def forward(self, x):
        if x.shape[-1] == x.numel():
            outshape = list(x.shape)
            y = self.bias.clone()
            outshape[-1] = self.bias.numel()
            dtype = x.dtype
            if self.faster:
                x = x.half()
                quant_cuda.vecquant3matmul_faster(x, self.qweight, y, self.scales, self.zeros)
            else:
                x = x.float()
                quant_cuda.vecquant3matmul(x, self.qweight, y, self.scales, self.zeros)
            y = y.to(dtype)
            return y.reshape(outshape)
        raise ValueError('Only supports a single token currently.')

def make_quant3(module, names, name='', faster=False):
    if isinstance(module, Quant3Linear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(
                module, attr, Quant3Linear(tmp.in_features, tmp.out_features, faster=faster)
            )
    for name1, child in module.named_children():
        make_quant3(child, names, name + '.' + name1 if name != '' else name1, faster=faster)

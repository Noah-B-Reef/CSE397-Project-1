# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import torch
from scipy.stats import truncexpon

r_jup_mean = 6991100000.0

mean_std_Rpl = torch.tensor([1.05, 0.95])
mean_std_a = torch.tensor([7.1, 0.35])
mean_std_alpha = torch.cat([torch.linspace(-18.0, -6.0, 100, dtype=torch.float32).unsqueeze(0), torch.linspace(4.5, 4.5, 100, dtype=torch.float32).unsqueeze(0)], dim=0)

if torch.cuda.is_available():
    dev = torch.device('cuda')
else:
    dev = torch.device("cpu")

def r_isotherm(a, r_P0):
    P0 = 0.01
    pressures = torch.logspace(-6, 2, 100).unsqueeze(0)
    return r_P0 / (1 + ((a/r_P0)*torch.log(pressures/P0)))

def get_radii(inputs, norm=True):
    R_pl = ((inputs[:,2] * mean_std_Rpl[1]) + mean_std_Rpl[0]) * r_jup_mean
    a = 10**((inputs[:,3] * mean_std_a[1]) + mean_std_a[0])

    radii = r_isotherm(a.unsqueeze(1), R_pl.unsqueeze(1))
    
    if norm:
        norm_radii = (radii - radii[:,-1].unsqueeze(1)) / (radii[:,0].unsqueeze(1) - radii[:,-1].unsqueeze(1)) * 2 - 1
        
        return radii, norm_radii
    
    return radii

def gen_test_samples(n, key="transition12", smooth=False, neg=True, R02=False):
    
    x = torch.ones((n,1))
    y = torch.ones((n,1))
    if R02:
        R_pl = ((torch.rand((n,1)) * 1.8 + 0.2) - mean_std_Rpl[0]) / mean_std_Rpl[1]
    else:
        R_pl = torch.rand((n,1))
    a = torch.normal(0.0, 1.0, size=(n,1))
    
    if smooth:
        rand_nums1 = torch.ones(n,99) * 1.0 / 99 * 8
        rand_nums2 = torch.ones(n,99) * 2.0 / 99 * 8
    else:
        rand_bool1 = torch.randint(0, 2, size=(n, 99), dtype=torch.bool)
        rand_nums1 = ((rand_bool1*2-1)) * torch.tensor(truncexpon.rvs(3 * torch.e, loc=0.0, scale=1.0 / torch.e / 6, size=(n, 99)), dtype=torch.float32)
        rand_nums2 = -torch.tensor(truncexpon.rvs(3 * torch.e, loc=0.0, scale=1.0 / torch.e / 6, size=(n, 99)), dtype=torch.float32) + 1.01
        rand_nums1 = (rand_nums1 + 1.0) / 99 * 8
        rand_nums2 = (rand_nums2 + 1.0) / 99 * 8
    
    if key == "1":
        rand_grad = rand_nums1
        
    elif key == "2":
        rand_grad = rand_nums2
        
    elif key == "random-switch":
        rand_bool2 = torch.randint(0, 2, size=(n, 99), dtype=torch.bool)
        rand_grad = torch.zeros((n,99), dtype=torch.float32)
        rand_grad[rand_bool2] = rand_nums1[rand_bool2]
        rand_grad[~rand_bool2] = rand_nums2[~rand_bool2]
        
        if neg:
            rand_grad[rand_grad < (0.8/99*8)] = (torch.rand(rand_grad[rand_grad < (0.8/99*8)].shape)*1.3 - 0.5) / 99 * 8
            
    elif key == "random-uniform":
        rand_grad = (torch.rand((n,99)) + 1.0) / 99 * 8
        
    elif key == "transition12" or key == "transition21":
        rand_grad = torch.cat([rand_nums1.unsqueeze(0), rand_nums2.unsqueeze(0)], dim=0)
    
    if key == "transition12" or key == "transition21":
        alphas = torch.zeros((2, n, 100))
        alphas[:, :, 49] = torch.rand((n,), dtype=torch.float32) * (15.0) - 20.0
        alphas[:, :, :49] = alphas[:,:,49].unsqueeze(-1) - torch.cumsum(rand_grad[:,:,:49], dim=-1).flip(-1)
        alphas[:, :, 50:] = alphas[:,:,49].unsqueeze(-1) + torch.cumsum(rand_grad[:,:,49:], dim=-1)
        
        alphas = torch.log10((10**alphas[0,:,:] + 10**alphas[1,:,:])) - 0.5
        
        if key == "transition21":
            alphas = (alphas - mean_std_alpha[0]) / mean_std_alpha[1]
            alphas = (-alphas * mean_std_alpha[1]) + mean_std_alpha[0] - 0.5
        
    else:
        alphas = torch.zeros((n, 100))
        alphas[:, 49] = torch.rand((n,), dtype=torch.float32) * (15.0) - 20.0
        alphas[:, :49] = alphas[:,49][:, None] - torch.cumsum(rand_grad[:,:49], dim=-1).flip(-1)
        alphas[:, 50:] = alphas[:,49][:, None] + torch.cumsum(rand_grad[:,49:], dim=-1)
        
    alphas = (alphas - mean_std_alpha[0]) / mean_std_alpha[1]
        
    inputs = torch.cat([x,y,R_pl,a,alphas], dim=-1)
    
    return inputs

def calc_transm(model, inputs, calc_prt=False, mask=False, rayleigh=False):
    
    radii, ys = get_radii(inputs)
    ind = 4
    alpha_ind = 4
    
    if type(rayleigh) is not bool:
        inputs = torch.cat([inputs[:,:2],torch.ones(inputs.shape[0]).unsqueeze(1)*0.0,inputs[:,2:4],torch.tensor(rayleigh).repeat(inputs.shape[0],1),inputs[:,4:]], dim=-1)
        alpha_ind = 6
    
    transm = []
    for i in range(100):
        inputs[:,1] = ys[:,i]
        transm.append(model(inputs.to(dev)).to(torch.device("cpu")).detach())            
    transm = torch.cat(transm, axis=-1)
    
    if mask:
        mask = inputs[:,alpha_ind:] <= -1.5
        transm[mask] = 1.0

        transm[transm > 1.0] = 1.0
        transm[transm < 0.0] = 0.0
    
    if calc_prt:
        alphas = 10**((inputs[:,alpha_ind:] * mean_std_alpha[1]) + mean_std_alpha[0])
        alphas[:,1:] = alphas[:,:-1] + alphas[:,1:]
        
        n = alphas.shape[0]
        diffS = torch.zeros((n,100,100))
        
        Rik = radii.reshape(n,1,100)**2 - radii.reshape(n,100,1)**2
        Rik[Rik<0.] = 0.
        Rik = torch.sqrt(Rik)
        diffS[:,1:,1:] = - Rik[:, 1:, 1:] + Rik[:, 1:, :-1]
        
        prt_transm = torch.einsum('ij,ikj->ik', alphas, diffS)
        prt_transm = torch.exp(-prt_transm)
        
        return transm, prt_transm
    
    return transm


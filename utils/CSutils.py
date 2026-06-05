# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import sys
from torch import nn

##################################
# Objective Functions
##################################
# Cross-Entropy Loss
NLL_loss = torch.nn.NLLLoss().cuda()
def Cross_Entropy(prob,lab):
    CE_loss = NLL_loss(torch.log(prob+1e-4), lab)
    return CE_loss


# Entropy Loss
def Entropy(prob):
    num_sam = prob.shape[0]
    Entropy = -(prob.mul(prob.log()+1e-4)).sum()
    return Entropy/num_sam


def compute_sigma(H):
    dists = torch.pdist(H)
    sigma = dists.median()/2
    return sigma.detach()


def ema_normalize(tensor, alpha=0.9):
    """Apply exponential moving average normalization to the input tensor."""
    mean = torch.zeros_like(tensor[0])
    var = torch.zeros_like(tensor[0])
    for i in range(tensor.size(0)):
        mean = alpha * mean + (1 - alpha) * tensor[i]
        var = alpha * var + (1 - alpha) * (tensor[i] - mean) ** 2
    std = torch.sqrt(var + 1e-8)
    return (tensor - mean) / std


class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, eps=1e-8):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type
        self.eps = eps

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples + self.eps)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / (bandwidth_temp + self.eps)) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.shape[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

def GaussianMatrix(X,Y,sigma, if_use_cdist=False, median_sigma=False):
    X = X.float()  # Convert to float
    Y = Y.float()  # Convert to float
    if not if_use_cdist:
        size1 = X.size()
        size2 = Y.size()
        G = (X*X).sum(-1)
        H = (Y*Y).sum(-1)
        Q = G.unsqueeze(-1).repeat(1,size2[0])
        R = H.unsqueeze(-1).T.repeat(size1[0],1)
        #print(G.shape, R.shape, X.shape, Y.shape, Q.shape, R.shape)
        H = Q + R - 2*X@(Y.T)
    else:
        H = torch.cdist(X, Y, p=2)**2

    if sigma > 0:
        H = torch.exp(-H/2/sigma**2)
    else:
        if median_sigma:
            sigma = compute_sigma(H)
            H = torch.exp(-H/2/sigma**2)
        else:
            sigma = H.mean().detach()
            H = torch.exp(-H/sigma)
    return H


def compute_median_sigma(x1, x2):
    """Compute the median of pairwise distances between samples in x1 and x2."""
    dists = torch.cdist(x1, x2, p=2)
    median_sigma = torch.median(dists).item()
    return median_sigma


def MKernel(x1, x2, sigmas=None, median_sigma=False):
    """Compute the Gaussian kernel matrix with multiple sigmas."""
    if median_sigma:
        base_sigma = compute_median_sigma(x1, x2)
        if sigmas is None:
            sigmas = [base_sigma * (2 ** i) for i in range(-2, 2)]  # Example range: [0.25*base_sigma, 0.5*base_sigma, base_sigma, 2*base_sigma, 4*base_sigma]
    else:
        if sigmas is None:
            sigmas = [1.0]  # Default sigma value if not provided

    K = 0
    for sigma in sigmas:
        K += GaussianMatrix(x1, x2, sigma)
    return K / len(sigmas)


# def GaussianMatrix(X, Y, sigma):
#     """Compute the Gaussian kernel matrix."""
#     X = X.float()
#     Y = Y.float()
#     size1 = X.size()
#     size2 = Y.size()
#     G = (X * X).sum(-1)
#     H = (Y * Y).sum(-1)
#     Q = G.unsqueeze(-1).repeat(1, size2[0])
#     R = H.unsqueeze(-1).T.repeat(size1[0], 1)
#     H = Q + R - 2 * X @ (Y.T)
#     H = torch.exp(-H / (2 * sigma ** 2))
#     return H


# CKB loss
# def CondCSD(x1,x2,y1,y2,sigma =5, if_use_cdist=False, median_sigma=False): # conditional cs divergence
#     # Input: N x d
#
#     # x1 = torch.tensor(x1)
#     # x2 = torch.tensor(x2)
#     # y1 = torch.tensor(y1)
#     # y2 = torch.tensor(y2)
#
#     K1 = GaussianMatrix(x1,x1,sigma, if_use_cdist, median_sigma) # a lot of 0 (1560)
#     K2 = GaussianMatrix(x2,x2,sigma, if_use_cdist, median_sigma) # 1560 0
#
#     L1 = GaussianMatrix(y1,y1,sigma, if_use_cdist, median_sigma)
#     L2 = GaussianMatrix(y2,y2,sigma, if_use_cdist, median_sigma)
#
#     #print(x1.shape, x2.shape, y1.shape, y2.shape, K1.shape, K2.shape, L1.shape, L2.shape)
#
#     K12 = GaussianMatrix(x1,x2,sigma, if_use_cdist, median_sigma) # nan happens  1600 0 ---> all zeros --> makes the later part nan
#     L12 = GaussianMatrix(y1,y2,sigma, if_use_cdist, median_sigma) #
#
#     K21 = GaussianMatrix(x2,x1,sigma, if_use_cdist, median_sigma) # nan happens  1600 0
#     L21 = GaussianMatrix(y2,y1,sigma, if_use_cdist, median_sigma)
#
#     # K1 = MK(x1, x1)
#     # K2 = MK(x2, x2)
#     # L1 = MK(y1, y1)
#     # L2 = MK(y2, y2)
#     # K12 = MK(x1, x2)
#     # L12 = MK(y1, y2)
#     # K21 = MK(x2, x1)
#     # L21 = MK(y2, y1)
#
#     H1 = K1*L1 # 1560 0
#     self_term1 = (H1.sum(-1)/((K1.sum(-1))**2)).sum(0) #
#
#     H2 = K2*L2
#     self_term2 = (H2.sum(-1)/((K2.sum(-1))**2)).sum(0)
#
#     ##################################DEBUG#################################################
#     H3 = K12*L12
#     cross_term1 = (H3.sum(-1)/((K1.sum(-1))*(K12.sum(-1)))).sum(0) # # nan first happens
#     ##################################DEBUG################################################
#     H4 = K21*L21
#     cross_term2 = (H4.sum(-1)/((K2.sum(-1))*(K21.sum(-1)))).sum(0)
#
#     cs1 = -2*torch.log(cross_term1) + torch.log(self_term1) + torch.log(self_term2)
#     cs2 = -2*torch.log(cross_term2) + torch.log(self_term1) + torch.log(self_term2)
#
#
#     return ((cs1+cs2)/2)
#
# def CS(x1,x2,sigma = 5, if_use_cdist=False, median_sigma=False): # conditional cs divergence
#     #x1 = torch.tensor(x1)
#     #x2 = torch.tensor(x2)
#
#     K1 = GaussianMatrix(x1,x1,sigma, if_use_cdist, median_sigma)
#     K2 = GaussianMatrix(x2,x2,sigma, if_use_cdist, median_sigma)
#
#     K12 = GaussianMatrix(x1,x2,sigma, if_use_cdist, median_sigma)
#     # K1 = MK(x1, x1)
#     # K2 = MK(x2, x2)
#     # K12 = MK(x1, x2)
#
#     dim1 = K1.shape[0]
#     self_term1 = K1.sum()/(dim1**2)
#
#     dim2 = K2.shape[0]
#     self_term2 = K2.sum()/(dim2**2)
#
#     cross_term = K12.sum()/(dim1*dim2)
#
#     cs = -2*torch.log(cross_term) + torch.log(self_term1) + torch.log(self_term2)
#
#     return cs


def CondCSD(x1, x2, y1, y2, sigmas=[5,6,7,8],  median_sigma=False):
    K1 = MKernel(x1, x1, sigmas, median_sigma)
    K2 = MKernel(x2, x2, sigmas, median_sigma)
    L1 = MKernel(y1, y1, sigmas, median_sigma)
    L2 = MKernel(y2, y2, sigmas, median_sigma)

    K12 = MKernel(x1, x2, sigmas, median_sigma)
    L12 = MKernel(y1, y2, sigmas, median_sigma)
    K21 = MKernel(x2, x1, sigmas, median_sigma)
    L21 = MKernel(y2, y1, sigmas, median_sigma)

    H1 = K1 * L1
    self_term1 = (H1.sum(-1) / (K1.sum(-1)**2)).sum(0)

    H2 = K2 * L2
    self_term2 = (H2.sum(-1) / (K2.sum(-1)**2)).sum(0)

    H3 = K12 * L12
    cross_term1 = (H3.sum(-1) / (K1.sum(-1) * K12.sum(-1))).sum(0)

    H4 = K21 * L21
    cross_term2 = (H4.sum(-1) / (K2.sum(-1) * K21.sum(-1))).sum(0)

    cs1 = -2 * torch.log(cross_term1) + torch.log(self_term1) + torch.log(self_term2)
    cs2 = -2 * torch.log(cross_term2) + torch.log(self_term1) + torch.log(self_term2)

    return (cs1 + cs2) / 2


def CS(x1, x2, sigmas=[5,6,7,8], median_sigma=False):
    K1 = MKernel(x1, x1, sigmas, median_sigma)
    K2 = MKernel(x2, x2, sigmas, median_sigma)
    K12 = MKernel(x1, x2, sigmas, median_sigma)

    dim1 = K1.shape[0]
    self_term1 = K1.sum() / (dim1**2)

    dim2 = K2.shape[0]
    self_term2 = K2.sum() / (dim2**2)

    cross_term = K12.sum() / (dim1 * dim2)
    return -2 * torch.log(cross_term) + torch.log(self_term1) + torch.log(self_term2)



def JointKDE_KL(
    z_source,
    z_target,
    yhat_source,
    yhat_target,
    sigmas_z=None,
    sigmas_y=None,
    median_sigma=False,
    eps=1e-8,
):
    """Estimate forward KL divergence KL(p_s || q_t) via joint KDE.

    The joint density over latent features and predicted probabilities is
    approximated with a separable Gaussian kernel K((z, ŷ), (z', ŷ')) = Kz(z, z') * Ky(ŷ, ŷ').

    Args:
        z_source (Tensor): Source latent features of shape [N_s, d].
        z_target (Tensor): Target latent features of shape [N_t, d].
        yhat_source (Tensor): Source predicted probabilities of shape [N_s, K].
        yhat_target (Tensor): Target predicted probabilities of shape [N_t, K].
        sigmas_z (List[float] | None): Bandwidths for feature kernel. Defaults to [5,6,7,8] when None.
        sigmas_y (List[float] | None): Bandwidths for label kernel. Defaults to [5,6,7,8] when None.
        median_sigma (bool): If True, use median heuristic to scale bandwidths.
        eps (float): Small constant for numerical stability inside logs.

    Returns:
        Tensor: Scalar tensor estimating KL(p_s || q_t) on the current mini-batch.
    """
    # Default bandwidth sets mirror CS/MKernel defaults when not provided
    if sigmas_z is None:
        sigmas_z = [5, 6, 7, 8]
    if sigmas_y is None:
        sigmas_y = [5, 6, 7, 8]

    # Joint kernel evaluations: Kz * Ky
    # Evaluate densities at source samples (forward KL averages over source batch)
    Kz_ss = MKernel(z_source, z_source, sigmas_z, median_sigma)
    Ky_ss = MKernel(yhat_source, yhat_source, sigmas_y, median_sigma)
    H_ss = Kz_ss * Ky_ss  # [N_s, N_s]

    Kz_st = MKernel(z_source, z_target, sigmas_z, median_sigma)
    Ky_st = MKernel(yhat_source, yhat_target, sigmas_y, median_sigma)
    H_st = Kz_st * Ky_st  # [N_s, N_t]

    # KDE estimates at each source evaluation point: mean over reference set
    ps_est = H_ss.mean(dim=1)  # p_s(u) for u in source batch
    qt_est = H_st.mean(dim=1)  # q_t(u) for u in source batch

    # Numerically stable log-ratio averaging
    kl_terms = torch.log(ps_est + eps) - torch.log(qt_est + eps)
    return kl_terms.mean()


def SymmetricJointKDE_KL(
    z_source,
    z_target,
    yhat_source,
    yhat_target,
    sigmas_z=None,
    sigmas_y=None,
    median_sigma=False,
    eps=1e-8,
):
    """Symmetric version: 0.5 * (KL(p_s || q_t) + KL(q_t || p_s)).

    This can be useful when a symmetric discrepancy is desired for fairness
    against symmetric measures like CS.
    """
    kl_st = JointKDE_KL(
        z_source,
        z_target,
        yhat_source,
        yhat_target,
        sigmas_z=sigmas_z,
        sigmas_y=sigmas_y,
        median_sigma=median_sigma,
        eps=eps,
    )
    kl_ts = JointKDE_KL(
        z_target,
        z_source,
        yhat_target,
        yhat_source,
        sigmas_z=sigmas_z,
        sigmas_y=sigmas_y,
        median_sigma=median_sigma,
        eps=eps,
    )
    return 0.5 * (kl_st + kl_ts)

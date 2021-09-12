import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as scio
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from time import *

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation."""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def NormalizeFea(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array((features**2).sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def L2_distance(Z):
    Z = Z.astype('float64')
    num = np.size(Z,0)
    AA = np.sum(Z**2,axis=1)
    AB = np.matmul(Z,Z.T)
    A = np.tile(AA,(num,1))
    B = A.T
    d = A + B -2*AB
    return d

def load_data(dataFile, k, sigma): # {multi-view}
    """Load Multi-view data."""
    print('Reading multi-view data...')
    data = scio.loadmat(dataFile)
    flag = 0
    i = 0
    for key in data :
        i = i + 1
        if flag == 0 and (type(data[key]) is np.ndarray) and i==(len(data)-1):
            features = data[key][0]
            flag = 1
            continue
        if flag == 1 and (type(data[key]) is np.ndarray) and i==len(data):
            labels = data[key]
    for i in range(0, len(features)):
        try:
            features[i] = preprocess_features(features[i].todense().getA())
        except:
            features[i] = preprocess_features(features[i])
    Dist = []  #distance adj
    for vIndex in range(0, len(features)):
        TempvData = features[vIndex]
        NorTempvData = NormalizeFea(TempvData)
        tempDM = L2_distance(NorTempvData)
        Dist.append(tempDM)
    Sim = []
    for ii in range(0, len(Dist)):
        Sim.append(sparse_mx_to_torch_sparse_tensor(normalize_adj(bs_convert2sim_knn(Dist[ii], k, sigma))))
    return Sim, labels

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def spectral(W, k):
    """
    SPECTRUAL spectral clustering
    :param W: Adjacency matrix, N-by-N matrix
    :param k: number of clusters
    :return: data point cluster labels, n-by-1 vector.
    """
    w_sum = np.array(W.sum(axis=1)).reshape(-1)
    D = np.diag(w_sum)
    _D = np.diag((w_sum + np.finfo(float).eps)** (-1 / 2))
    L = D - W
    L = _D @ L @ _D
    eigval, eigvec = np.linalg.eig(L)
    eigval_argsort = eigval.real.astype(np.float32).argsort()
    F = np.take(eigvec.real.astype(np.float32), eigval_argsort[:k], axis=-1)
    idx = KMeans(n_clusters=k).fit(F).labels_
    return idx

def bestMap(L1,L2):
    '''
    bestmap: permute labels of L2 to match L1 as good as possible
        INPUT:  
            L1: labels of L1, shape of (N,) vector
            L2: labels of L2, shape of (N,) vector
        OUTPUT:
            new_L2: best matched permuted L2, shape of (N,) vector
    version 1.0 --December/2018
    Modified from bestMap.m (written by Deng Cai)
    '''
    if L1.shape[0] != L2.shape[0] or len(L1.shape) > 1 or len(L2.shape) > 1: 
        raise Exception('L1 shape must equal L2 shape')
        return 
    Label1 = np.unique(L1)
    nClass1 = Label1.shape[0]
    Label2 = np.unique(L2)
    nClass2 = Label2.shape[0]
    nClass = max(nClass1,nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        for j in range(nClass2):
            G[j,i] = np.sum((np.logical_and(L1 == Label1[i], L2 == Label2[j])).astype(np.int64))
    c,t = linear_sum_assignment(-G)
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[t[i]]
    return newL2

def bs_convert2sim_knn(dist, K, sigma):
    dist = dist/np.max(np.max(dist, 1))
    sim = np.exp(-dist**2/(sigma**2))
    if K>0:
        idx = sim.argsort()[:,::-1]
        sim_new = np.zeros_like(sim)
        for ii in range(0, len(sim_new)):
            sim_new[ii, idx[ii,0:K]] = sim[ii, idx[ii,0:K]]
        sim = (sim_new + sim_new.T)/2
    else:
        sim = (sim + sim.T)/2
    return sim

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

def evaluate(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    acc = np.sum(label == pred)/pred.shape[0]
    ari = metrics.adjusted_rand_score(label, pred)
    f = metrics.fowlkes_mallows_score(label, pred)
    Precision = metrics.precision_score(label, pred, average='macro')
    Recall = metrics.recall_score(label, pred, average='macro')
    Purity = purity_score(label, pred)
    return nmi, acc, ari, f, Precision, Recall, Purity

import torch
import torch.nn as nn

class DSSNE(nn.Module):
    def __init__(self, n, I, mu, alpha):
        super(DSSNE, self).__init__()
        self.mu = mu
        self.n = n
        self.alpha = 1.0/(1.0+alpha)
        self.A = nn.Parameter(torch.FloatTensor(n,n))
        self.A.data = I
        self.I = I
    def forward(self, W):        
        A_tmp = list()
        loss = 0
        for w in W:
            w = w.to_dense()
            h = w.sum(1)
            d_inv = 1.0/torch.sqrt(h)
            DD = torch.unsqueeze(d_inv,dim=-1)*d_inv
            S = w*DD
            h = torch.mm(torch.mm(S, self.A), S.t())
            loss += self.mu*torch.mean(h*h)
            A_tmp.append(h)
        A = self.alpha*sum(A_tmp, 0) + (1-self.alpha)*self.I
        for w in A_tmp:
            h = w.sum(0)
            d = torch.sqrt(h)
            D = torch.diag(d)
            d_inv = 1.0/d            
            DD = torch.unsqueeze(d_inv,dim=-1)*d_inv
            w = w*DD
            w = w + w.t()
            loss += torch.mm(A.t(), torch.mm((D-w),A)).trace()/self.n
        return sum(A_tmp, 0), A_tmp, loss
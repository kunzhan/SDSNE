import numpy as np
import torch
import argparse
from utils import load_data, spectral, bestMap, evaluate
from sklearn.cluster import KMeans
import math
from model import DSSNE
from time import *
import warnings
warnings.filterwarnings('ignore')

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='3sources', help = 'Dataset for multi-view training.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stop.')
parser.add_argument('--cuda', type=int, default=1, help='Use CUDA for training.')
parser.add_argument('--train', type=int, default=1, help='Train or not.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
parser.add_argument('--wd', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--knn', type=int, default=16, help='KNN K.')
parser.add_argument('--sigma', type=float, default=0.5, help='Weight parameters for knn.')
parser.add_argument('--mu', type=float, default=0.769, help='Weight parameters for L2.')
parser.add_argument('--alpha', type=float, default=0.769, help='Weight parameters for diffusion.')
parser.add_argument('--sc', type=int, default=1, help='Use spectral cluster or k-means')

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

if args.sc:
    print('_____________'+args.dataset+' Spectral Cluster'+'__________________')
else:
    print('_____________'+args.dataset+' K-means'+'__________________')
print("Parameter settings:",args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# training params
nb_epochs = args.epochs
patience = args.patience
lr = args.lr
l2_coef = args.wd
para_mu = args.mu
para_alpha = args.alpha
Train_flag = args.train

begin_time = time()
dataFile = '/home/kzhan/data/' + args.dataset + '.mat'
adj, labels = load_data(dataFile, args.knn, args.sigma)
kmeansK = len(np.unique(labels))
Y = labels.T[0]
end_time = time()
run_time = end_time - begin_time
print('Reading Time:{}s'.format(run_time))

n_nodes = len(Y) #nodes
ViewN = len(adj)  #views
cnt_wait = 0
best = 1e9
best_t = 0

I = torch.eye(n_nodes)
if args.cuda:
    print('Using GPU', end='  ')
    I = I.cuda()
    for i in range(0, ViewN):
        adj[i] = adj[i].cuda()
else:
    print('Using CPU', end='  ')

model = DSSNE(n_nodes, I, para_mu, para_alpha)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
if args.cuda:
    model.cuda()
if Train_flag:
    for epoch in range(nb_epochs):
        adj_hat, adj_view, loss = model(adj)
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), './log/'+args.dataset+'_best.pkl')
        else:
            cnt_wait += 1
        if cnt_wait == patience or math.isnan(loss):
            print('\nEarly stopping!', end='')
            break
        loss.backward()
        optimiser.step()
        #print("\r\rMulti-view Learning: Epoch:%04d || " % epoch, "Loss:%5.4f || " % loss, \
        #    "Best:%04d || " % best_t, "Wait:%03d" % (cnt_wait+1), end=' ')
    print('  Loading {}th epoch'.format(best_t))

model.load_state_dict(torch.load('./log/'+args.dataset+'_best.pkl'))
A, adj_view, l= model(adj)
A = A + A.t()
A = A.cpu().detach().numpy()
if args.sc:
    predY = spectral(A, kmeansK)
else:
    predY = KMeans(n_clusters=kmeansK).fit(A).labels_
gnd_Y = bestMap(predY, Y)
nmi, acc, ari, f1, precision, recall, purity = evaluate(gnd_Y, predY)
end_time = time()
run_time = end_time - begin_time
print('Result: NMI:%1.4f || ACC:%1.4f || ARI:%1.4f || F-score:%1.4f || Precision:%1.4f \
|| Recall:%1.4f || Purity:%1.4f || Time:%5.4fs'%(nmi, acc, ari, f1, precision, recall, purity, run_time))

print('&{:.3f}'.format(nmi)+'$\pm${:.3f}'.format(0.0)+
        '&{:.3f}'.format(acc)+'$\pm${:.3f}'.format(0.0)+
        '&{:.3f}'.format(ari)+'$\pm${:.3f}'.format(0.0)+
        '&{:.3f}'.format(f1)+'$\pm${:.3f}'.format(0.0)+
        '&{:.3f}'.format(precision)+'$\pm${:.3f}'.format(0.0)+
        '&{:.3f}'.format(recall)+'$\pm${:.3f}'.format(0.0)+
        '&{:.3f}'.format(purity)+'$\pm${:.3f}\\\\'.format(0.0))
if args.sc:
    print('_____________'+args.dataset+' Spectral Cluster'+'__________________\n\n\n')
else:
    print('_____________'+args.dataset+' K-means'+'__________________\n\n\n')
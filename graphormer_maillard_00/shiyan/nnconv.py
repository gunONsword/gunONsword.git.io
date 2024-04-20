import copy, os, pickle, random, re, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU

import torch_geometric.transforms as T
from torch_geometric.nn import NNConv, global_add_pool
from torch_geometric.data import DataLoader
# from torch_geometric.datasets import QM9
from QM9G16 import QM9G16


# from torch_geometric.data import (InMemoryDataset, download_url, extract_zip, Data)

def dumpData(data1):
  print(
    data1.edge_attr,  #  last element bond length in Angstrom
    data1.edge_index, 
    data1.idx,
    data1.keys, 
    data1.name,
    data1.pos, 
    data1.x, #nodes
    data1.y, 
    data1.z
  )

#计算数据集中目标变量的均值和标准差，并进行线性回归拟合
def calcMeanStd(dataset):
  from sklearn.linear_model import LinearRegression
#  count number of atoms
  ndata = len(dataset)
  X = np.zeros((ndata,5),dtype=np.float32)    #生成0矩阵
  y = np.zeros(ndata,dtype=np.float32)
  for idata,data1 in enumerate(dataset[0:]):
    X[idata,:] = np.sum(data1.x[:,0:5].numpy(),axis=0)
#    X[idata,:] = np.sum(data1.x[:,0:5],axis=0)
    y[idata] = data1.y[0,target]
  reg = LinearRegression(fit_intercept=False).fit(X, y)  #reg会包含训练好的参数
  dy = y - reg.predict(X)  #损失函数
  print('linear regression',reg.score(X, y), reg.coef_, reg.intercept_, dy.mean(),dy.std())
  return reg.coef_[:],dy.std()


def gnn_model_summary(model):      #用于打印神经网络模型的参数信息，包括每个层的参数名称、参数的形状（shape）和参数的数量 GNN
    model_params_list = list(model.named_parameters())
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer.Parameter", "Param Tensor Shape", "Param #")
    print(line_new)
    print("----------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0] 
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>20}  {:>25} {:>15}".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("----------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)


class MyTransform(object):  #转换object的类型
  def __call__(self, data):
    xsum = data.x[:,0:5].sum(axis=0)
    data.y = (data.y[:, target] - sum(xsum * mean)) / std
    return data

transform = T.Compose([MyTransform()]) #创建数据转换序列，可以方便地对图数据进行一系列预处理或增强操作

Restart = False
target = 12  #  total energy
dim = 64
dimnn = 16
BondKind = 4  #  four kinds of bonds
dataset = QM9G16(root='/home/yasudak/data/QM9')
nL1 = 4  #  int(m1.group(1))
nL2 = 2  #  int(m1.group(2))
jobname = 'nnconv_%d%d_%dx%d' % (dim,dimnn,nL1,nL2)
print("Start! hidden dims, # layers =",jobname)
imol = 0; dumpData(dataset[imol])

# split data to tran/val/test
if Restart==True:  #  restart
  model_pth = jobname+'.model'
  checkpoint = torch.load(model_pth)
  rand_perm = checkpoint['rand_perm']
else:
  rand_perm = [i for i in range(len(dataset))]
  random.shuffle(rand_perm)

# Normalize targets to mean~0, std~1.
mean,std = calcMeanStd(dataset[rand_perm[10000:]])
print('mean,std,target=',mean,std,target)

dataset = QM9G16(root='/home/yasudak/data/QM9', transform=transform)
imol = 0; dumpData(dataset[imol])


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin0 = torch.nn.Linear(dataset.num_features, dim)
        nn1 = Sequential(Linear(BondKind, dimnn*dimnn))
        self.gru0 = GRU(dimnn, dim)
        self.gru1 = GRU(dimnn, dim)
        self.gru2 = GRU(dimnn, dim)
        self.gru3 = GRU(dimnn, dim)
#        self.gru4 = GRU(dimnn, dim)
#        self.gru5 = GRU(dimnn, dim)
#        self.grus = [self.gru0,self.gru1,self.gru2,self.gru3,self.gru4,self.gru5][:nL1]
        self.grus = [self.gru0,self.gru1,self.gru2,self.gru3][:nL1]
        self.conv1 = NNConv(dimnn, dimnn, nn1, aggr='mean')
        self.lin1_64_16 = torch.nn.Linear(dim, dimnn)
#        self.lin1 = torch.nn.Linear(dim, dim)
        self.lin_64_1 = torch.nn.Linear(dim, 1, bias=False)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for j in range(nL1):   #nL1是GRU的层数       NNConv 层进行卷积操作，用于处理图的边信息
          gru = self.grus[j]
          for i in range(nL2):
            out1 = self.lin1_64_16(out)
            out2 = self.conv1(out1, data.edge_index, data.edge_attr)
            out, h = gru(out2.unsqueeze(0), h); out = out.squeeze(0)

        out = self.lin_64_1(out)
        out = global_add_pool(out, data.batch)
        return out.view(-1)


def train():
  model.train()
  loss_all = 0
  for data in train_loader:
    data = data.to(device)
    optimizer.zero_grad()
    loss = criterion(model(data), data.y)
    loss.backward()
    loss_all += loss.item() * data.num_graphs
    optimizer.step()
  return loss_all / len(train_loader.dataset)


def test(loader):
  model.eval()
  error = 0
  for data in loader:
    data = data.to(device)
    error += (model(data) * std - data.y * std).abs().sum().item()  # MAE 平均绝对误差
  return error / len(loader.dataset)


def test1(loader):
  model.eval()
  errorlist = []
  for data in loader:
    names =  data.name
    xsum = data.x[:,0:5].sum(axis=0)
    y0 = sum(xsum * mean).item()
    ys = data.y.to('cpu').detach().numpy() * std + y0
    data = data.to(device)
    preds = model(data).to('cpu').detach().numpy() * std + y0
    for l1 in zip(names,ys,preds):
      errorlist.append(l1)
  return errorlist


# Split datasets.
test_dataset = dataset[rand_perm[:10000]]
val_dataset = dataset[rand_perm[10000:20000]]
train_dataset = dataset[rand_perm[20000:]]
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for itry in range(0,1): # (0,3):

  model = Net().to(device)
  gnn_model_summary(model)

  pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print('# model params=',pytorch_total_params)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  criterion = torch.nn.MSELoss()
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.00001)
  model.train()
  best_val_error = None
#  for epoch in range(1, 301):
  for epoch in range(1, 3):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train()
    train_error = test(train_loader)
    val_error = test(val_loader)
    scheduler.step(val_error)

    if best_val_error is None or val_error <= best_val_error:
        test_error = test(test_loader)
        best_val_error = val_error

    print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Train MAE: {:.7f}, Validation MAE: {:.7f}, '
          'Test MAE: {:.7f}'.format(epoch, lr, loss, train_error, val_error, test_error), flush=True)

  model.eval()
  errorlist = test1(test_loader)
  print(errorlist[0:10])
  with open(jobname+'_%d_err.pickle' % itry,'wb') as f1:
    pickle.dump(errorlist,f1)
  
  model = model.to('cpu')
  model_pth = jobname+'_%d.model' % itry
  torch.save({
    'rand_perm': rand_perm,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, model_pth)



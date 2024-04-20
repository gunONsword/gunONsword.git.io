import os
import torch
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip, Data)
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.datasets import QM9

import pickle

def saveDebugData(dataset,filename):
  savedata = []
  for data in dataset: # [0:64]:
    edge_attr = data.edge_attr.detach().numpy().copy()
    edge_index = data.edge_index.detach().numpy().copy()
    idx = data.idx.detach().numpy().copy()
    pos = data.pos.detach().numpy().copy()
    x = data.x.detach().numpy().copy()
    y = data.y.detach().numpy().copy()
    z = data.z.detach().numpy().copy()
    savedata.append([edge_attr,edge_index,idx,data.name,pos,x,y,z])
  print('saveDebugData',savedata[0],savedata[-1],len(savedata))
  with open(filename,'wb') as f1:
    pickle.dump(savedata,f1)


class QM9G16(InMemoryDataset):

  raw_url = 'file:/home/yasudak/data/QM9/raw/qm9_v2.pt'

  def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
    super(QM9G16, self).__init__(root, transform, pre_transform, pre_filter)
    self.data, self.slices = torch.load(self.processed_paths[0])

  def mean(self, target):
    y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
    return y[:, target].mean().item()

  def std(self, target):
    y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
    return y[:, target].std().item()

  @property
  def raw_file_names(self):
    return ['qm9_v2.pt']

  @property
  def processed_file_names(self):
    return ['qm9_g16.pt']

  def download(self):
    file_path = download_url(self.raw_url, self.raw_dir)

  def process(self):
#    print('should not come here')
#    raise RuntimeError

    with open('/home/yasudak/data/QM9/raw/qm9_g16.pickle','rb') as f1:
      savedata = pickle.load(f1)
    data_list = []
    for edge_attr,edge_index,idx,name,pos,x,y,z in savedata:
      edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
      edge_index = torch.tensor(edge_index, dtype=torch.int64)
      idx = torch.tensor(idx, dtype=torch.int64)
      pos = torch.tensor(pos, dtype=torch.float32)
      x = torch.tensor(x, dtype=torch.float32)
      y = torch.tensor(y, dtype=torch.float32)
      z = torch.tensor(z, dtype=torch.float32)
      data1 = Data(x=x, z=z, pos=pos, edge_index=edge_index,
        edge_attr=edge_attr, y=y, name=name, idx=idx)
      data_list.append(data1)
    torch.save(self.collate(data_list), self.processed_paths[0])

if __name__ == '__main__':
  qm9data = QM9G16(root='/home/yasudak/data/QM9')
  print(qm9data[0])
#  saveDebugData(qm9data,'/home/yasudak/data/QM9/raw/qm9_g16.pickle')


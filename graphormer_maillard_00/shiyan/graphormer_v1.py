from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


listElem = [1,6,7,8,16]
dictElem = {k:i for i,k in enumerate(listElem)}


def MyTransform(mean,std,x,y):
  x = torch.tensor(x, dtype=torch.int32)
#  xsum = np.array([sum(x==z) for z in listElem])
  xsum = np.array([sum(x==z) for z in range(len(listElem))])  #x中z出现的次数
  y = (y - sum(xsum * mean)) / std
#  print(x,xsum,y); raise RuntimeError
  return y


import numpy as np
import torch
device=torch.device('cuda')

import pickle
from datasets import Dataset, DatasetDict
datapth = '/home/wangzp/data/QM9'
with open(datapth+'/processed/maillard_v0.pickle','rb') as f1:
  mean,std = pickle.load(f1)


#生成新文件
prep_data = False
if prep_data: # not os.path.exists(datapth):
  with open(datapth+'/raw/dataset_230720.pickle','rb') as f1:
    savedata = pickle.load(f1)
  data_dict = {'edge_index':[], 'edge_attr':[], 'y':[], 'num_nodes':[], 'node_feat':[], 'name':[]}
  for x,edge_index,edge_attr,y,name in savedata:
    x = [dictElem[x1] for x1 in x]  #  atomic_num to serial_num
    y = MyTransform(mean,std,x,y)  
    x = [[x1] for x1 in x]  #  atomic_num to serial_num
#    for l1 in edge_attr:
#      l1[1] = 0; l1[2] = 0  #  clear HB bondlength, angle
    for k1 in range(len(edge_attr)):
      edge_attr[k1] = edge_attr[k1][0]
#    print('edge_attr=',type(edge_attr),edge_attr); raise RuntimeError
    data_dict['edge_attr'].append(edge_attr)
    data_dict['edge_index'].append(edge_index)
    data_dict['y'].append([y])
    data_dict['num_nodes'].append(len(x))
    data_dict['node_feat'].append(x)
    data_dict['name'].append('%s_%s' % (name[0],name[2][0]))
  ds1 = Dataset.from_dict(data_dict)

  train_testvalid = ds1.train_test_split(test_size=0.2)
  test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
  ds2 = DatasetDict({
    'train': train_testvalid['train'],    #训练集
    'test': test_valid['test'],           #测试集
    'validation': test_valid['train']})   #验证集   8:1:1
  print(ds2,ds2['train'][0])
  ds2.save_to_disk(datapth)


from datasets import load_from_disk
dataset = load_from_disk(datapth)
print(dataset,dataset['train'][0])

from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator
datapth = '/home/wangzp/data/QM9/processed'  #预处理

if prep_data: # not os.path.exists(datapth):
  dataset_processed = dataset.map(preprocess_item, batched=False)
  dataset_processed.save_to_disk(datapth)
dataset_processed = load_from_disk(datapth)
print(dataset_processed,dataset_processed['train'][0])


from transformers import (
#    AutoModel,
    GraphormerConfig,
    GraphormerForGraphClassification,
    GraphormerModel,
#    GraphormerCollator
)

config = GraphormerConfig(
  num_classes=1,
  num_atoms=32*1, # GraphormerCollator
  num_edges=32*1, # GraphormerCollator
  num_spatial=8, # algos_graphormer
  num_edge_dis=16,
  num_in_degree=16,
  num_out_degree=16,
  spatial_pos_max=20,
  multi_hop_max_dist=5,
  edge_type="multi_hop",
  max_nodes=64,
  share_input_output_embed=False,
#  num_layers=8,
  embedding_dim=32,
  ffn_embedding_dim=32,
  num_attention_heads=4,
  self_attention=True,
  activation_fn="gelu",
  dropout=0.0,
  attention_dropout=0.1,
  layerdrop=0.0,
  bias=True,
  embed_scale=None,
  num_trans_layers_to_freeze=0,
  encoder_normalize_before=True,
  pre_layernorm=False,
  apply_graphormer_init=True,
  freeze_embeddings=False,
  q_noise=0.0,
  qn_block_size=8,
  kdim=None,
  vdim=None,
  traceable=False,
  pad_token_id=0,
  bos_token_id=1,
  eos_token_id=2,
  )
model = GraphormerForGraphClassification(config)  #  initialize model
model = GraphormerForGraphClassification.from_pretrained("./graphormer_maillard")  #  load previous weight
# from transformers import AutoModel
# model = AutoModel.from_pretrained("clefourrier/graphormer-base-pcqm4mv1")
# model = GraphormerForGraphClassification.from_pretrained(
#      "clefourrier/pcqm4mv2_graphormer_base",
#      num_classes=2, # num_classes for the downstream task
#      ignore_mismatched_sizes=True,
#  )
# print(model,sum(p.numel() for p in model.parameters()))


# import networkx as nx
# import matplotlib.pyplot as plt
# graph = dataset["train"][0]
# edges = graph["edge_index"]
# num_edges = len(edges[0])
# num_nodes = graph["num_nodes"]
# G = nx.Graph()
# G.add_nodes_from(range(num_nodes))
# G.add_edges_from([(edges[0][i], edges[1][i]) for i in range(num_edges)])
# nx.draw(G)


from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(
    "graph-regression",
    logging_dir="graph-regression",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    auto_find_batch_size=True,
    gradient_accumulation_steps=1, # 2,
    dataloader_num_workers=0, #1, 
    num_train_epochs=100,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    push_to_hub=False,
    dataloader_pin_memory=False,
#    fp16=True,
)

# import numpy as np
# import evaluate
# metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
#    with open('tmp.pickle','wb') as f1:
#      pickle.dump([logits,labels],f1)
    mae = abs(logits[0][:,0] - labels).sum()/len(labels)
#    mae *= std  #  to atomic unit
    return {'mae':mae}

#训练
trainer = Trainer(
    model=model.to(device),
    args=training_args,
    train_dataset=dataset_processed["train"],
    eval_dataset=dataset_processed["validation"],
    data_collator=GraphormerDataCollator(),
    compute_metrics=compute_metrics,
)
# print_gpu_utilization()

train_results = trainer.train()
print_summary(train_results)

# from torch.profiler import profile, record_function, ProfilerActivity
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#   with record_function("model_inference"):
#     train_results = trainer.train()
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# raise RuntimeError

#  GraphormerDataCollator() does not support trainer.predict?
# test_dataset = dataset_processed["test"],
# predictions,labels,metrics = trainer.predict(test_dataset) # , metric_key_prefix="predict")  #保存权重
# print(predictions,labels,metrics)
trainer.save_model("./graphormer_maillard")
# trainer.push_to_hub()


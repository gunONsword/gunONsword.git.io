from pynvml import *

def print_gpu_utilization():  #输出GPUliyonglv
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


import numpy as np
import torch
device=torch.device('cuda')


from transformers import (
#    AutoModel,
    GraphormerConfig,
    GraphormerForGraphClassification,
    GraphormerModel,
#    GraphormerCollator
)

#参数集
config = GraphormerConfig(
  num_classes=2,
  num_atoms=32*9, # GraphormerCollator
  num_edges=32*3, # GraphormerCollator
  num_spatial=32, # algos_graphormer
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
model = GraphormerForGraphClassification(config)
# model = GraphormerForGraphClassification.from_pretrained("./graphormer_small")
# from transformers import AutoModel
# model = AutoModel.from_pretrained("clefourrier/graphormer-base-pcqm4mv1")
# model = GraphormerForGraphClassification.from_pretrained(
#     "clefourrier/pcqm4mv2_graphormer_base",
#     num_classes=2, # num_classes for the downstream task 
#     ignore_mismatched_sizes=True,
# )
print(model,sum(p.numel() for p in model.parameters()))

from datasets import load_dataset
dataset = load_dataset("OGB/ogbg-molhiv")
print(dataset,dataset['train'][0])
dataset = dataset.shuffle(seed=0)
 
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

from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator
dataset_processed = dataset.map(preprocess_item, batched=False)
# dataset_processed["train"] = dataset_processed["train"].shuffle(seed=42).select(range(1000))


from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(
    "molhiv",
    logging_dir="molhiv",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    auto_find_batch_size=True,
    gradient_accumulation_steps=2,
    dataloader_num_workers=0, #1, 
    num_train_epochs=20,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    push_to_hub=False,
    dataloader_pin_memory=False,
    fp16=True,
)

# import numpy as np
# import evaluate
# metric = evaluate.load("accuracy")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model.to(device),
    args=training_args,
    train_dataset=dataset_processed["train"],
    eval_dataset=dataset_processed["validation"],
    data_collator=GraphormerDataCollator(),
#    compute_metrics=compute_metrics,
)
print_gpu_utilization()

train_results = trainer.train()
print_summary(train_results)
trainer.save_model("./molhiv")
# trainer.push_to_hub()


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from dgl import load_graphs

from get_adj import get_second_directed_adj
from model import DiGCN_Inception_Block_Ranking, SAGE
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, classification_report

import warnings
warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

graph, label_dict = load_graphs("./data.bin", [0])
graph = graph[0].to(device)
node_features = graph.ndata['feat'].to(device)
node_labels = graph.ndata['label'].type(torch.LongTensor).to(device)
train_mask = graph.ndata['train_mask'].to(device)
valid_mask = graph.ndata['val_mask'].to(device)
test_mask = graph.ndata['test_mask'].to(device)
test_mask = torch.logical_or(test_mask, valid_mask)
n_features = node_features.shape[1]
n_labels = int(node_labels.max().item() + 1)


def evaluate(model, graph, features, labels, mask, method_name):
    mask = mask.to(torch.bool)
    model.eval()
    with torch.no_grad():
        if method_name == 'SAGE':
            logits = model(graph, features).to(device)
        else:
            edge_index = graph.edges()
            edge_weights = torch.FloatTensor(graph.edata['weight'])
            edge_index1 = edge_index.clone().to(device)
            edge_weights1 = edge_weights.clone().to(device)
            edge_index2, edge_weights2 = get_second_directed_adj(edge_index,features.shape[0],
                                                                 features.dtype,
                                                                 edge_weights)
            edge_index2 = edge_index2.to(device)
            edge_weights2 = edge_weights2.to(device)
            edge_index = (edge_index1, edge_index2)
            edge_weights = (edge_weights1, edge_weights2)
            del edge_index2, edge_weights2
            logits = model(edge_index,edge_weights, features).to(device)
        logits = logits[mask]
        labels = labels[mask]

        # print(torch.argmax(logits, 1).shape)
        # print(labels.shape)
        # _, indices = torch.max(logits, dim=1)
        # correct = torch.sum(indices == labels)
        # return correct.item() * 1.0 / len(labels)
        y_pred = torch.argmax(logits, 1).cpu().numpy()
        y_true = labels.cpu().numpy()
        print(y_pred[0], y_true[0])
        # , multi_class="ovr")
        return f1_score(y_true, y_pred, average="macro")


def final_classification_report(model, graph, features, labels, mask, method_name):
    mask = mask.to(torch.bool)
    model.eval()
    with torch.no_grad():
        if method_name == 'SAGE':
            logits = model(graph, features).to(device)
        else:
            edge_index = graph.edges()
            edge_weights = torch.FloatTensor(graph.edata['weight'])
            edge_index1 = edge_index.clone().to(device)
            edge_weights1 = edge_weights.clone().to(device)
            edge_index2, edge_weights2 = get_second_directed_adj(edge_index, features.shape[0],
                                                                 features.dtype,
                                                                 edge_weights)
            edge_index2 = edge_index2.to(device)
            edge_weights2 = edge_weights2.to(device)
            edge_index = (edge_index1, edge_index2)
            edge_weights = (edge_weights1, edge_weights2)
            del edge_index2, edge_weights2
            logits = model(edge_index, edge_weights, features).to(device)
        logits = logits[mask]
        labels = labels[mask]

        # print(torch.argmax(logits, 1).shape)
        # print(labels.shape)
        # _, indices = torch.max(logits, dim=1)
        # correct = torch.sum(indices == labels)
        # return correct.item() * 1.0 / len(labels)
        y_pred = torch.argmax(logits, 1).cpu().numpy()
        y_true = labels.cpu().numpy()
        print(y_pred[0], y_true[0])
        target_names = ['Blackhole', 'Flooding',
                        'Grayhole', 'Normal', 'Scheduling']
        return classification_report(y_true, y_pred, target_names=target_names, digits=3, output_dict=True)


method_name = 'DiGCN_Inception_Block_Ranking'
# method_name = "SAGE"
EPOCH = 500
if method_name == 'SAGE':
    model = SAGE(in_feats=n_features, hid_feats=100,
                 out_feats=n_labels).to(device)
else:
    model = DiGCN_Inception_Block_Ranking(
        num_features=n_features, embedding_dim=32, out=n_labels, dropout=0.2).to(device)
opt = torch.optim.Adam(model.parameters())
train_mask = train_mask.to(torch.bool).to(device)
hist_train_loss, hist_test_loss, hist_train_f1, hist_test_f1=[], [], [], []
for epoch in range(EPOCH):
    model.train()
    # forward propagation by using all nodes
    if method_name == 'SAGE':
        logits = model(graph, node_features).to(device)
    else:
        edge_index = graph.edges()
        edge_weights = torch.FloatTensor(graph.edata['weight'])
        edge_index1 = edge_index.clone().to(device)
        edge_weights1 = edge_weights.clone().to(device)
        edge_index2, edge_weights2 = get_second_directed_adj(edge_index, node_features.shape[0],
                                                             node_features.dtype,
                                                             edge_weights)
        edge_index2 = edge_index2.to(device)
        edge_weights2 = edge_weights2.to(device)
        edge_index = (edge_index1, edge_index2)
        edge_weights = (edge_weights1, edge_weights2)
        del edge_index2, edge_weights2
        logits = model(edge_index, edge_weights, node_features).to(device)
    # compute loss
    # train loss
    loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])
    print("Train Loss:", loss.item())
    hist_train_loss.append(loss.item())
    # test loss
    loss_test = F.cross_entropy(logits[test_mask], node_labels[test_mask])
    print("Test Loss:", loss_test.item())
    hist_test_loss.append(loss_test.item())
    # compute validation accuracy
    # train f1
    train_f1 = evaluate(model, graph, node_features,
                        node_labels, train_mask,  method_name)
    print("Train F1-score:", train_f1)
    hist_train_f1.append(train_f1)
    # test f1
    test_f1 = evaluate(model, graph, node_features,
                       node_labels, test_mask, method_name)
    print("Test F1-score:", test_f1)
    hist_test_f1.append(test_f1)
    # backward propagation
    opt.zero_grad()
    loss.backward()
    opt.step()

# test_f1 = evaluate(model, graph, node_features, node_labels, test_mask, method_name)
# print("Test F1-score:", test_f1)

print("Train dataset classification report")
report=final_classification_report(model, graph, node_features, node_labels, train_mask, method_name)
df_report=pd.DataFrame(report).transpose()
df_report.to_csv("train_classification_report.csv")

print("Test dataset classification report")
report=final_classification_report(model, graph, node_features, node_labels, test_mask, method_name)
df_report=pd.DataFrame(report).transpose()
df_report.to_csv("test_classification_report.csv")

np.savez("hist_train_test.npz",
          hist_train_loss=hist_train_loss,
            hist_test_loss=hist_test_loss,
              hist_train_f1=hist_train_f1,
                hist_test_f1=hist_test_f1)



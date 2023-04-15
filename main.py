from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from dgl import load_graphs

from model import SAGE_FCL, DiGCN_Inception_Block_Ranking, SAGE
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix

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
node_labels = graph.ndata['label'].to(device).type(torch.long)
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
        if method_name == 'SAGE' or method_name == 'SAGE_FCL':
            logits = model(graph, features).to(device)
        else:
            logits = model(graph.edges(), (graph.edata['weight'], graph.edata['weight']), node_features).to(device)
        logits = logits[mask]
        labels = labels[mask]

        # print(torch.argmax(logits, 1).shape)
        # print(labels.shape)
        # _, indices = torch.max(logits, dim=1)
        # correct = torch.sum(indices == labels)
        # return correct.item() * 1.0 / len(labels)
        y_pred = torch.argmax(logits, 1).cpu().numpy()
        y_true = labels.cpu().numpy()
        # , multi_class="ovr")
        return f1_score(y_true, y_pred, average="macro")


def final_classification_report(model, graph, features, labels, mask, method_name,):
    mask = mask.to(torch.bool)
    model.eval()
    with torch.no_grad():
        if method_name == 'SAGE' or method_name == 'SAGE_FCL':
            logits = model(graph, features).to(device)
        else:
            logits = model(graph.edges(), (graph.edata['weight'], graph.edata['weight']), node_features).to(device)
        logits = logits[mask]
        labels = labels[mask]

        # print(torch.argmax(logits, 1).shape)
        # print(labels.shape)
        # _, indices = torch.max(logits, dim=1)
        # correct = torch.sum(indices == labels)
        # return correct.item() * 1.0 / len(labels)
        y_pred = torch.argmax(logits, 1).cpu().numpy()
        y_true = labels.cpu().numpy()
        target_names = ['Blackhole', 'Flooding', 'Grayhole', 'Normal', 'Scheduling']
        np.savez("y_pred_true.npz", y_true=y_true, y_pred=y_pred, target_names=target_names)

        cm = confusion_matrix(y_true, y_pred)
        tp = np.diagonal(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (tp + fp + fn)
        # Compute TPR and FPR for each class
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        fnr = fn / (tp + fn)
        tnr = tn / (tn + fp)
        # Compute accuracy
        accuracy = (tp+tn)/(tp+fp+fn+tn)
        all ={}
        for i in range(len(tpr)):
            all[target_names[i]]={"tpr":tpr[i],"fpr":fpr[i],"fnr":fnr[i],"tnr":tnr[i],"accuracy":accuracy[i]}
        print("True Positives:", tp)
        print("True Negatives:", tn)
        print("False Positives:", fp)
        print("False Negatives:", fn)
        print("Accuracy:", accuracy)
        cm=classification_report(y_true, y_pred, target_names=target_names, digits=3, output_dict=True)
        for i,j in cm.items():
            try:
                cm[i]["tpr"] =all[i]["tpr"]
                cm[i]["fpr"] = all[i]["fpr"]
                cm[i]["fnr"] = all[i]["fnr"]
                cm[i]["tnr"] = all[i]["tnr"]
                cm[i]["accuracy"] = all[i]["accuracy"]
            except:
                pass
        return cm


# method_name = 'DiGCN_Inception_Block_Ranking'
method_name = "SAGE"
EPOCH = 5
if method_name == 'SAGE':
    model = SAGE(in_feats=n_features, hid_feats=128, hid2_feats=64,
                 out_feats=n_labels).to(device)
elif method_name == 'SAGE_FCL':
    model = SAGE_FCL(in_feats=n_features, hid_feats=128, hid2_feats=64,
                 out_feats=n_labels).to(device)
else:
    model = DiGCN_Inception_Block_Ranking(
        num_features=n_features, embedding_dim=32, out=n_labels, dropout=0.2).to(device)
#opt = torch.optim.Adam(model.parameters(), lr=0.001)
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
#opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
train_mask = train_mask.to(torch.bool).to(device)
hist_train_loss, hist_test_loss, hist_train_f1, hist_test_f1=[], [], [], []
for epoch in range(EPOCH):
    #if epoch == int(EPOCH * (2/3)):
    #    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    # forward propagation by using all nodes
    if method_name == 'SAGE' or method_name == 'SAGE_FCL':
        logits = model(graph, node_features).to(device)
    else:
        logits = model(graph.edges(), (graph.edata['weight'], graph.edata['weight']), node_features).to(device)
    # compute loss
    # train loss
    #loss = F.nll_loss(F.log_softmax(logits[train_mask], dim=1), node_labels[train_mask])
    loss = F.cross_entropy(F.log_softmax(logits[train_mask], dim=1), node_labels[train_mask])
    print("Train Loss:", loss.item())
    hist_train_loss.append(loss.item())
    # test loss
    #loss_test = F.nll_loss(F.log_softmax(logits[test_mask], dim=1), node_labels[test_mask])
    loss_test = F.cross_entropy(F.log_softmax(logits[test_mask], dim=1), node_labels[test_mask])
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
print(df_report)

print("Test dataset classification report")
report=final_classification_report(model, graph, node_features, node_labels, test_mask, method_name)
df_report=pd.DataFrame(report).transpose()
df_report.to_csv("test_classification_report.csv")
print(df_report)

np.savez("hist_train_test.npz",
          hist_train_loss=hist_train_loss,
            hist_test_loss=hist_test_loss,
              hist_train_f1=hist_train_f1,
                hist_test_f1=hist_test_f1)



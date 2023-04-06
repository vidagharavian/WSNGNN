import torch
from dgl import load_graphs

from model import DiGCN_Inception_Block_Ranking, SAGE
import torch.nn.functional as F
graph, label_dict = load_graphs("./data.bin", [0])
graph=graph[0]
node_features = graph.ndata['feat']
node_labels = graph.ndata['label'].type(torch.LongTensor)
train_mask = graph.ndata['train_mask']
valid_mask = graph.ndata['val_mask']
test_mask = graph.ndata['test_mask']
n_features = node_features.shape[1]
n_labels = int(node_labels.max().item() + 1)

def evaluate(model, graph, features, labels, mask,method_name):
    model.eval()
    with torch.no_grad():
        if method_name == 'SAGE':
            logits = model(graph, features)
        else:
            logits = model(graph.edges(),(graph.edata['weight'],graph.edata['weight']),features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

#method_name ='DiGCN_Inception_Block_Ranking'
method_name="SAGE"
if method_name == 'SAGE':
    model = SAGE(in_feats=n_features, hid_feats=100, out_feats=n_labels)
else:
    model = DiGCN_Inception_Block_Ranking(num_features=n_features,embedding_dim=32,out=n_labels,dropout=0.2)
opt = torch.optim.Adam(model.parameters())

for epoch in range(50):
    model.train()
    # forward propagation by using all nodes
    if method_name =='SAGE':
        logits = model(graph,node_features)
    else:
        logits = model(graph.edges(),(graph.edata['weight'],graph.edata['weight']),node_features)
    # compute loss
    loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])
    # compute validation accuracy
    acc = evaluate(model, graph, node_features, node_labels, valid_mask,method_name)
    print("accuracy: "+str(acc))
    # backward propagation
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("loss: "+str(loss.item()))
acc = evaluate(model, graph, node_features, node_labels, test_mask,method_name)
print("test accuracy: "+str(acc))
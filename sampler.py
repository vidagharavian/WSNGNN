import dgl
import numpy as np
import torch.nn.functional as F
import torch
from dgl import load_graphs

from model import SAGE


def load_data_set():
    return load_graphs("./data.bin",[0])


def add_reverse(dataset):
    graph = dataset[0]
    graph = dgl.add_reverse_edges(graph)
    node_features = graph.ndata['feat']
    num_features = node_features.shape[1]
    num_classes = (graph.ndata['label'].max() + 1).item()
    return int(num_classes),int(num_features),graph

def split_data_train_test(graph,device):
    train_nids = torch.flatten(torch.nonzero(graph.ndata['train_mask'].to(device)))
    valid_nids = torch.flatten(torch.nonzero(graph.ndata['val_mask'].to(device)))
    test_nids = torch.flatten(torch.nonzero(graph.ndata['test_mask'].to(device)))
    return train_nids,valid_nids,test_nids

def neighber_sampling(graph,train_nids,valid_nids,device):
    sampler = dgl.dataloading.NeighborSampler([4, 4])
    train_dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to DGL's DataLoader.
        graph,  # The graph
        train_nids,  # The node IDs to iterate over in minibatches
        sampler,  # The neighbor sampler
        device=device,  # Put the sampled MFGs on CPU or GPU
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=1024,  # Batch size
        shuffle=True,  # Whether to shuffle the nodes for every epoch
        drop_last=False,  # Whether to drop the last incomplete batch
        num_workers=0  # Number of sampler processes
    )
    valid_dataloader = dgl.dataloading.DataLoader(
        graph, valid_nids, sampler,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        device=device
    )
    return train_dataloader,valid_dataloader
import tqdm
import sklearn.metrics
def run_model():
    import warnings
    warnings.filterwarnings('ignore')

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    best_accuracy = 0
    best_model_path = 'model.pt'
    graph, node_labels = load_graphs("./data.bin", [0])
    num_classes,num_features,graph=add_reverse(graph)
    train_nids, valid_nids, test_nids = split_data_train_test(graph,device)
    train_dataloader,valid_dataloader = neighber_sampling(graph,train_nids,valid_nids,device)
    model = SAGE(num_features, 128, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    for epoch in range(100):
        model.train()

        with tqdm.tqdm(train_dataloader) as tq:
            for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                # feature copy from CPU to GPU takes place here
                inputs = mfgs[0].srcdata['feat']
                labels = (mfgs[-1].dstdata['label']).to(torch.int64)

                predictions = model(mfgs, inputs)

                loss = F.cross_entropy(predictions, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()

                accuracy = sklearn.metrics.accuracy_score(labels.cpu().numpy(),
                                                          predictions.argmax(1).detach().cpu().numpy())

                tq.set_postfix({'loss': '%.03f' % loss.item(), 'acc': '%.03f' % accuracy}, refresh=False)

        model.eval()

        predictions = []
        labels = []
        with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
            for input_nodes, output_nodes, mfgs in tq:
                inputs = mfgs[0].srcdata['feat']
                labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
                predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)
            accuracy = sklearn.metrics.accuracy_score(labels, predictions)
            print('Epoch {} Validation Accuracy {}'.format(epoch, accuracy))
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), best_model_path)



run_model()
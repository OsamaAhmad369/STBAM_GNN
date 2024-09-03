import sys
sys.path.append('')
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
from src.utils.dataloader import DataLoaderCreator
from src.utils.base import BaseEngine
from src.utils.args import get_public_config
import argparse
import numpy as np



class GNN(nn.Module):
    def __init__(self,num_classes,nodes):
        super(GNN, self).__init__()
        self.nodes=nodes
        self.num_classes = num_classes
        self.conv1 = GATConv(67, 16,heads=2,dropout=0.1)
        self.conv2 = GATConv(32, 16,heads=2,dropout=0.1)
        self.conv3 = GATConv(32, 16,heads=2,dropout=0.1)
        self.out1 = nn.Linear(192*32, num_classes)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, x, adj,indices):
        x_graph=torch.zeros((adj.shape[0],3*self.nodes,32)).to(x.device)
        for i in range(adj.shape[0]):
            weighted_adj_matrix = adj[i,:,:]
            coo_matrix = sp.coo_matrix(weighted_adj_matrix.detach().cpu().numpy())  # Convert to Scipy COO matrix
            edge_index, edge_attr = from_scipy_sparse_matrix(coo_matrix)
            edge_index=edge_index.to(x.device)
            edge_attr=edge_attr.to(x.device)
            x_n=x[i,:,:]
            x_n = F.relu(self.conv1(x_n,edge_index,edge_attr=edge_attr))
            x_n = F.relu(self.conv2(x_n,edge_index,edge_attr=edge_attr))
            x_n = F.relu(self.conv3(x_n,edge_index,edge_attr=edge_attr))
            x_graph[i,:,:]=x_n
        x_graph=x_graph.reshape(shape=(x_graph.shape[0],x_graph.shape[1]*x_graph.shape[2]))
        y=self.out1(x_graph)
        return y, None,None
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
def main():

    torch.autograd.set_detect_anomaly(True)
    seed = 42
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    classes=4
    
    
    
    parser = get_public_config()
    args = parser.parse_args()
    
    model = GNN(num_classes=classes, nodes=args.nodes).to(args.device)
    model.apply(initialize_weights)
    loss_fn=torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lrate)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model.params (M): ', num_params/1000000)
    
    dataloader_creator = DataLoaderCreator(batch_size=args.bs)
    train_loader, val_loader, test_loader = dataloader_creator.get_loaders()
    engine=BaseEngine(model,optimizer,loss_fn,args.penalty,args.device,train_loader,val_loader,test_loader,args.epochs, args.show_cm)
    
    if args.mode=="train":
        engine.train()
    else:
        engine.evaluateTest(args.weightspath)

if __name__ == "__main__":
    main()    
    



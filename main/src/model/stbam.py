import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
from src.utils.dataloader import DataLoaderCreator
from src.utils.base import BaseEngine
from src.utils.args import get_public_config
from src.utils.transformer import PositionalEncoding, EncoderLayer
import argparse



class GNN(nn.Module):
    def __init__(self,num_classes,nodes,num_heads,dmodel):
        super(GNN, self).__init__()
        self.nodes=nodes
        self.num_classes = num_classes
        self.d_model = dmodel
        max_seq_length=self.nodes*3
        dropout=0.1
        self.num_heads = num_heads
        num_layers=1
        d_ff=2048

        self.feature_up_sampling = nn.Linear(67, self.nodes*3)
        self.linear_layer = nn.Linear(max_seq_length, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, max_seq_length)
        self.output_mapping_layer=nn.Linear(self.d_model,max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(self.d_model, self.num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=0.1)  # Dropout layer

        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.bn3 = torch.nn.BatchNorm1d(32)

        self.conv1 = GATConv(67, 16,heads=2,dropout=0.1)
        self.conv2 = GATConv(32, 16,heads=2,dropout=0.1)
        self.conv3 = GATConv(32, 16,heads=2,dropout=0.1)

        self.out1 = nn.Linear(192*32, num_classes)
        self.softmax=nn.Softmax(dim=1)
    def forward(self, x, adj,indices):

        x_up=self.feature_up_sampling(x)
        adj_x=adj+x_up

        adj_x=self.dropout(self.positional_encoding(self.linear_layer(adj_x)))
        enc_output = adj_x

        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, mask=None)
        enc_output = self.output_mapping_layer(enc_output)
        madj=enc_output
    
        madj_S = (madj + madj.transpose(1,2))/2
        madj_AS = (madj - madj.transpose(1,2))/2
        nadj=F.relu(adj+madj_S)         #skip connection
        # Convert each adjacency matrix in the batch to an edge index
        x_graph=torch.zeros((nadj.shape[0],3*self.nodes,32)).to(x.device)
        for i in range(nadj.shape[0]):
                weighted_adj_matrix = nadj[i,:,:]
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
        return y, nadj,madj_AS

def get_config():
    parser = get_public_config()
    parser.add_argument('--head',type=int,default=8)
    parser.add_argument('--dmodel',type=int,default=512)
    args = parser.parse_args()
    return args

def initialize_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
def main():
    classes=4   
    torch.autograd.set_detect_anomaly(True)
    seed = 42
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    args = get_config()
    model = GNN(num_classes=classes,nodes=args.nodes,num_heads=args.head,dmodel=args.dmodel).to(args.device)
    model.apply(initialize_weights)
    loss_fn=torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lrate)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model.params (M): ', num_params/1000000)

    dataloader_creator = DataLoaderCreator(batch_size=args.bs)
    train_loader, val_loader, test_loader = dataloader_creator.get_loaders()
    engine=BaseEngine(model,optimizer,loss_fn,args.penalty,args.device,train_loader,val_loader,test_loader,args.epochs)

    if args.mode=="train":
            engine.train()
    else:
            engine.evaluateTest(args.weightspath)

if __name__ == "__main__":
    main()

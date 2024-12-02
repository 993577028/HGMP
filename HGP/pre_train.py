import torch
import copy
import torch.optim as optim
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch.nn.functional as F
from random import shuffle
import random
import os.path as osp
from torch_geometric.utils import shuffle_node
from torch.nn import BCEWithLogitsLoss
from torch_geometric.data import HeteroData
import dgl
from scripts.constant import data_target
from torch_geometric.nn import  global_mean_pool,global_add_pool,global_max_pool

import torch_geometric.transforms as T
from torch_geometric.datasets import IMDB,HGBDataset
from torch_geometric.loader import HGTLoader
from HGP.prompt import HGNN,GCL_GCN,GAT
from HGP.utils import gen_ran_output,load_data4pretrain,mkdir, graph_views,graph_pool
import torch.nn as nn
from layers.discriminator import Discriminator
from layers.readout import AvgReadout
from scripts.pytorchtools import EarlyStopping

class GraphCL(torch.nn.Module):

    def __init__(self, hgnn, hid_dim=64):
        super(GraphCL, self).__init__()
        self.hgnn = hgnn
        self.projection_head = torch.nn.Sequential(torch.nn.Linear(hid_dim, hid_dim),
                                                   torch.nn.ReLU(inplace=True),
                                                   torch.nn.Linear(hid_dim, hid_dim))

    def forward_cl_hgt(self,targetnode, graph):
        x=graph.ndata['x']
        edge_index_dict = {}
        for etype in graph.canonical_etypes:
            edge_index = graph.edges(etype=etype)
            src = edge_index[0].unsqueeze(0)
            tar = edge_index[1].unsqueeze(0)
            edge_index = torch.cat((src, tar), dim=0)
            edge_index_dict[etype] = edge_index

        x_dict = self.hgnn(targetnode,x, edge_index_dict)
        x=graph_pool('mean',x_dict,graph)
        x = self.projection_head(x)
        return x

    def forward_cl_shgn(self,targetnode, graph):
        x=graph.ndata['x']
        edge_index_dict = {}
        for etype in graph.canonical_etypes:
            edge_index = graph.edges(etype=etype)
            src = edge_index[0].unsqueeze(0)
            tar = edge_index[1].unsqueeze(0)
            edge_index = torch.cat((src, tar), dim=0)
            edge_index_dict[etype] = edge_index
        #x=graph.x_dict

        x_dict = self.hgnn(targetnode,graph, x)
        x=graph_pool('mean',x_dict,graph)
        x = self.projection_head(x)
        return x
    def forward_cl_gcn(self,targetnode, graph):
        x = graph.ndata['x']
        x_dict=self.hgnn(graph,x)
        x = graph_pool('mean', x_dict, graph)
        x = self.projection_head(x)
        return x
    def forward_cl_gat(self,targetnode, graph):
        x = graph.ndata['x']
        x_dict=self.hgnn(graph,x,False)
        x = graph_pool('mean', x_dict, graph)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + 1e-4)
        loss = - torch.log(loss).mean() + 10
        return loss


class HDGI(nn.Module):
    def __init__(self,num_out, hgnn,device):
        super(HDGI, self).__init__()
        self.device=device
        self.hgnn = hgnn
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.disc = Discriminator(num_out)


    def forward(self,targetnode, sample1, sample2,msk, samp_bias1, samp_bias2):
        h_1 = self.hgnn(targetnode,sample1.x_dict,sample1.edge_index_dict)
        c = self.read(h_1, msk)
        c = self.sigm(c)
        h_2 = self.hgnn(targetnode,sample2.x_dict,sample2.edge_index_dict)
        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def loss_hdgi(self,logits,batch_size):
        lbl_1 = torch.ones(1,batch_size)
        lbl_2 = torch.zeros(1,batch_size)
        lbl = torch.cat((lbl_1, lbl_2), 1)
        lbl=lbl.to(self.device)
        logits=torch.unsqueeze(logits, 0)

        loss = BCEWithLogitsLoss()(logits, lbl)

        return loss

    def embed(self, seq, adjs, sparse, msk):
        h_1 = self.hgat(seq, adjs)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()


class PreTrain(torch.nn.Module):
    def __init__(self, args,ntypes,metadata,num_class,num_etypes,input_dims):
        super(PreTrain, self).__init__()
        self.pretext = args.pretext
        self.hgnn_type=args.hgnn_type
        self.device=args.device
        self.args=args
        if(args.hgnn_type=='GCN'):
            self.hgnn=GCL_GCN(None,input_dims,args.hidden_dim,num_class,args.num_layers,F.elu,args.dropout,self.hgnn_type)
        elif(args.hgnn_type=='GAT'):
            heads = [args.num_heads] * args.num_layers + [1]
            self.hgnn = GAT(None,input_dims,args.hidden_dim,num_class,args.num_layers,heads,F.elu,args.dropout,args.dropout,0.05,False,self.hgnn_type)
        else:
            self.hgnn = HGNN(hid_dim=args.hidden_dim,out_dim=args.hidden_dim,hgnn_type=self.hgnn_type,num_layer=args.num_layers,num_heads=args.num_heads,dropout=args.dropout,metadata=metadata,ntypes=ntypes,num_etypes=num_etypes,input_dims=input_dims,args=args)
        if args.pretext == 'HDGI':
            self.model = HDGI(num_out=args.hidden_dim,hgnn=self.hgnn,device=self.device)
        elif args.pretext in ['GraphCL', 'SimGRACE']:
            self.model = GraphCL(self.hgnn,hid_dim=args.hidden_dim)
        else:
            raise ValueError("pretext should be GraphCL, SimGRACE")

    def train_hdgi(self,model,graph,optimizer):

        model.train()
        train_loss_accum = 0
        total_step = 0
        g1, g2 = shuffle_nodes(graph, '0')
        for step, batch in enumerate(zip(g1, g2)):
            batch1, batch2 = batch
            optimizer.zero_grad()
            batch1, batch2=batch1.to(self.device), batch2.to(self.device)
            logits=model(self.targetnode,batch1,batch2,None,None,None)

            batch_size=batch1.num_nodes('0')
            loss=model.loss_hdgi(logits,batch_size)
            loss.backward()
            optimizer.step()

            train_loss_accum += float(loss.detach().cpu().item())
            total_step = total_step + 1

        return train_loss_accum / total_step

    def get_loader(self, graph_list, batch_size,
                   aug1=None, aug2=None, aug_ratio=None, pretext="GraphCL"):

        if len(graph_list) % batch_size == 1:
            raise KeyError(
                "batch_size {} makes the last batch only contain 1 graph, \n which will trigger a zero bug in GraphCL!")

        if pretext == 'GraphCL':
            shuffle(graph_list)
            if aug1 is None:
                aug1 = random.sample(['dropN', 'permE', 'maskN'], k=1)
            if aug2 is None:
                aug2 = random.sample(['dropN', 'permE', 'maskN'], k=1)
            if aug_ratio is None:
                aug_ratio = random.randint(2, 3) * 1.0 / 10  # 0.1,0.2,0.3
                #aug_ratio = 0.3

            print("===graph views: {} and {} with aug_ratio: {}".format(aug1, aug2, aug_ratio))

            view_list_1 = []
            view_list_2 = []
            for g in graph_list:
                view_g = graph_views(data=g, aug=aug1, aug_ratio=aug_ratio)
                view_list_1.append(view_g)
                view_g = graph_views(data=g, aug=aug2, aug_ratio=aug_ratio)
                view_list_2.append(view_g)

            dataloader = dgl.dataloading.GraphDataLoader
            # you must set shuffle=False !
            loader1 = dataloader(view_list_1, batch_size=batch_size, shuffle=False)
            loader2 = dataloader(view_list_2, batch_size=batch_size, shuffle=False)
            # loader1 = DataLoader(view_list_1, batch_size=batch_size, shuffle=False)
            # loader2 = DataLoader(view_list_2, batch_size=batch_size, shuffle=False)

            return loader1, loader2
        elif pretext == 'SimGRACE':
            #loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False, num_workers=1)
            dataloader = dgl.dataloading.GraphDataLoader
            loader=dataloader(graph_list, batch_size=batch_size, shuffle=False, num_workers=1)
            return loader, None  # if pretext==SimGRACE, loader2 is None
        else:
            raise ValueError("pretext should be GraphCL, SimGRACE")

    def train_simgrace(self, model, loader, optimizer,device):
        model.train()
        train_loss_accum = 0
        total_step = 0
        target = data_target[self.args.dataset]
        for step, data in enumerate(loader):
            optimizer.zero_grad()
            data = data.to(device)
            # batch_list = {node_type: data[node_type].batch.to(device)
            #                for node_type in data.node_types
            #                }
            #size=data.batch_size


            x2 = gen_ran_output(target,data, model)

            x1 = model.forward_cl_gcn(target,data)
            x2 = Variable(x2.detach().data.to(device), requires_grad=False)
            loss = model.loss_cl(x1, x2)
            loss.backward()
            optimizer.step()
            train_loss_accum += float(loss.detach().cpu().item())
            total_step = total_step + 1

        return train_loss_accum / total_step

    def train_graphcl(self, model, loader1, loader2, optimizer,batch_size,device):
        model.train()
        train_loss_accum = 0
        total_step = 0
        for step, batch in enumerate(zip(loader1, loader2)):
            batch1, batch2 = batch
            optimizer.zero_grad()
            target=data_target[self.args.dataset]

            # batch_list1={node_type:batch1[node_type].batch.to(device)
            #     for node_type in batch1.node_types
            # }
            # batch_list2 = {node_type: batch2[node_type].batch.to(device)
            #                for node_type in batch2.node_types
            #                }
            # size=batch1.batch_size
            if(self.hgnn_type=='HGT'):
                x1 = model.forward_cl_hgt(target, batch1.to(device))
                x2 = model.forward_cl_hgt(target, batch2.to(device))
            elif(self.hgnn_type=='SHGN'):
                x1 = model.forward_cl_shgn(target, batch1.to(device))
                x2 = model.forward_cl_shgn(target, batch2.to(device))
            elif(self.hgnn_type=='GCN'):
                x1 = model.forward_cl_gcn(target, batch1.to(device))
                x2 = model.forward_cl_gcn(target, batch2.to(device))
            elif (self.hgnn_type == 'GAT'):
                x1 = model.forward_cl_gat(target, batch1.to(device))
                x2 = model.forward_cl_gat(target, batch2.to(device))


            loss = model.loss_cl(x1, x2)

            loss.backward()
            optimizer.step()

            train_loss_accum += float(loss.detach().cpu().item())
            total_step = total_step + 1

        return train_loss_accum / total_step

    def train(self, graph_batch_size,node_batch_size,dataname, graph_list, lr=0.01,decay=0.0001, epochs=100,aug1='dropN',aug2='permE',seed=None,aug_ration=None):

        loader1, loader2 = self.get_loader(graph_list, graph_batch_size, aug1=aug1, aug2=aug2,aug_ratio=aug_ration,
                                           pretext=self.pretext)
        #print('start training {} | {} | {}...'.format(dataname, pre_train_method, gnn_type))
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)

        train_loss_min = 1000000
        early_stopping = EarlyStopping(patience=30,verbose=False,save_path="./pre_trained_hgnn/{}.{}.{}.hid{}.np{}.pth".format(dataname, self.pretext, self.hgnn_type,self.args.hidden_dim,self.args.num_samples))
        for epoch in range(1, epochs + 1):  # 1..100
            if self.pretext == 'HDGI':
                train_loss = self.train_hdgi(self.model,graph,optimizer)
            elif self.pretext == 'GraphCL':
                train_loss = self.train_graphcl(self.model,loader1,loader2,optimizer,node_batch_size,self.args.device)
            elif self.pretext == 'SimGRACE':
                train_loss = self.train_simgrace(self.model,loader1,optimizer,self.args.device)
            else:
                raise ValueError("pretext should be GraphCL, SimGRACE")

            early_stopping(train_loss,self.model.hgnn)
            if early_stopping.early_stop:
                #print('Early stopping!')
                break

            # print("***epoch: {}/{} | train_loss: {:.8}".format(epoch, epochs, train_loss))
            #
            # if train_loss_min > train_loss:
            #     train_loss_min = train_loss
            #     torch.save(self.model.hgnn.state_dict(),
            #                "./pre_trained_hgnn/{}.{}.{}.pth".format(dataname, self.pretext, self.hgnn_type))
            #     # do not use '../pre_trained_gnn/' because hope there should be two folders: (1) '../pre_trained_gnn/'  and (2) './pre_trained_gnn/'
            #     # only selected pre-trained models will be moved into (1) so that we can keep reproduction
            #     print("+++model saved ! {}.{}.{}.pth".format(dataname, self.pretext, self.hgnn_type))



def copy_heterodata(heteragraph):
    if isinstance(heteragraph, HeteroData):
        g = HeteroData()
        for node_type, node_store in heteragraph.node_items():
            for attr, value in node_store.items():
                g[node_type][attr] = value

        for edge_type, edge_store in heteragraph.edge_items():
            for attr, value in edge_store.items():
                g[edge_type][attr] = value

        return g

def shuffle_nodes(loader,targetnode):
    graph=loader
    loader1,loader2=list(),list()

    loader1.append(graph)
    nodes=graph.node_stores[0]['x']
    nodes,_ = shuffle_node(nodes)
    newgraph=copy_heterodata(graph)
    newgraph.set_value_dict('x',{targetnode:nodes
    })
    loader2.append(newgraph)

    return loader1,loader2

if __name__ == '__main__':
    print("PyTorch version:", torch.__version__)

    if torch.cuda.is_available():
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
        device = torch.device("cuda")
    else:
        print("CUDA is not available")
        device = torch.device("cpu")

    print(device)

    mkdir('../pre_trained_hgnn/')
    # do not use '../pre_trained_gnn/' because hope there should be two folders: (1) '../pre_trained_gnn/'  and (2) './pre_trained_gnn/'
    # only selected pre-trained models will be moved into (1) so that we can keep reproduction

    pretext = 'HDGI'

    dataname, num_parts, batch_size = 'IMDB', 64, 64

    hgnn_type='HGT'


    graph_list,targetnode = load_data4pretrain(dataname=dataname,num_sample=num_parts,batch_size=batch_size)

    graph=graph_list[0]
    # print(graph)
    # print(graph.x_dict)
    # print(graph.edge_index_dict)
    metadata = graph.metadata()
    ntypes = graph.node_types

    pt = PreTrain(pretext, hgnn_type,gln=2,metadata=metadata,ntypes=ntypes,hid_dim=256, targetnode=targetnode,device=device)

    pt.model.to(device)
    pt.train(dataname, graph_list, batch_size=10, lr=0.01, decay=0.0001,
             epochs=100)

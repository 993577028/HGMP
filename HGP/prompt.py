import torch
import torch.nn.functional as F
from torch_geometric.nn import  global_mean_pool, TransformerConv
from torch_geometric.data import Batch, Data
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from torch_geometric.nn import HGTConv, Linear
from dgl.nn.pytorch import GATConv
from openhgnn import SimpleHGN
import dgl
#hgprompt和run_gcn会用这个
class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims]) #创建一个全连接层列表，其中每个全连接层将输入特征的维度映射到隐藏层节点数量
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False,allow_zero_in_degree=True))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation,allow_zero_in_degree=True))
        # output layer
        self.predict=nn.Linear(num_hidden,num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph,features_list,featrue_indices,reverse_indices,flag):
        if(flag=='run'):
            keys = list(features_list.keys())#dgl重排后的节点类型顺序
            keys_index=[]
            sum=0
            for key in features_list:
                sum+=features_list[key].shape[0]
                keys_index.append(sum)
            reordered_keys = [keys[i] for i in featrue_indices]
            reorder_x_dict = {key: features_list[key] for key in reordered_keys}

            feats_emd=[]
            for key in reorder_x_dict:
                feats_emd.append(reorder_x_dict[key])

            h = []
            for fc, feature in zip(self.fc_list, feats_emd):
                h.append(fc(feature))
            #再改回dgl图内节点类型的排序
            h=[h[i] for i in reverse_indices]
            h = torch.cat(h, 0)
            for i, layer in enumerate(self.layers):
                h = self.dropout(h)
                h = layer(dgl.to_homogeneous(graph), h)

            res={}
            for i,key in enumerate(keys):
                if(i==0):
                    res[key]=h[0:keys_index[i]]
                else:
                    res[key] = h[keys_index[i-1]:keys_index[i]]

            return res
        elif(flag=='pretrain'):
            h = []
            for fc, feature in zip(self.fc_list, features_list):
                h.append(fc(feature))
            h = torch.cat(h, 0)
            for i, layer in enumerate(self.layers):
                h = self.dropout(h)
                h = layer(self.g, h)
            return h
        elif(flag=='GCN'):
            keys = list(features_list.keys())  # dgl重排后的节点类型顺序
            keys_index = []
            sum = 0
            for key in features_list:
                sum += features_list[key].shape[0]
                keys_index.append(sum)

            feats_emd = [features_list[key] for key in features_list]

            h = []
            for fc, feature in zip(self.fc_list, feats_emd):
                h.append(fc(feature))
            h = torch.cat(h, 0)
            for i, layer in enumerate(self.layers):
                h = self.dropout(h)
                h = layer(dgl.to_homogeneous(graph), h)
            h=self.predict(h)

            res = {}
            for i, key in enumerate(keys):
                if (i == 0):
                    res[key] = h[0:keys_index[i]]
                else:
                    res[key] = h[keys_index[i - 1]:keys_index[i]]
            return res

#demo用的是下面这个
class GCL_GCN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout,
                 gnn_type):
        super(GCL_GCN, self).__init__()
        self.hgnn_type=gnn_type
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims]) #创建一个全连接层列表，其中每个全连接层将输入特征的维度映射到隐藏层节点数量
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False,allow_zero_in_degree=True))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation,allow_zero_in_degree=True))
        # output layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph,features_list):
        keys = list(features_list.keys())
        keys_index = []
        sum = 0
        for key in features_list:
            sum += features_list[key].shape[0]
            keys_index.append(sum)

        feats_emd = [features_list[key] for key in features_list]

        h = []
        for fc, feature in zip(self.fc_list, feats_emd):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            homo_g=dgl.to_homogeneous(graph)
            homo_g=dgl.remove_self_loop(homo_g)
            homo_g=dgl.add_self_loop(homo_g)
            h = layer(homo_g, h)
        res = {}
        for i, key in enumerate(keys):
            if (i == 0):
                res[key] = h[0:keys_index[i]]
            else:
                res[key] = h[keys_index[i - 1]:keys_index[i]]
        return res

class HGT(torch.nn.Module):
    def __init__(self, node_types,metadata,hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)


        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata,
                           num_heads)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self,targetnode, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        x_dict={
            node_type : self.lin(x)
            for node_type,x in x_dict.items()
        }
        # node_embd=self.lin(x_dict[targetnode])

        return x_dict

class myshgn(torch.nn.Module):
    def __init__(self, edge_feats,num_etypes,input_dims,hid_dim,out_dim,num_layer,heads,dropout,ntypes,slope):
        super().__init__()

        self.fc_list = nn.ModuleList([nn.Linear(in_dim, hid_dim, bias=True) for in_dim in input_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        self.simplehgn = SimpleHGN(edge_feats,num_etypes,[hid_dim],hid_dim,out_dim,num_layer,heads,dropout,slope,True,0.05,ntypes)


    def forward(self,hg, x_dict):

        for fc, feature in zip(self.fc_list, x_dict):
            x_dict[feature]=(fc(x_dict[feature]))

        ans=self.simplehgn(hg,x_dict)

        return ans

class GAT(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 hgnn_type):
        super(GAT, self).__init__()
        self.g = g
        self.hgnn_type=hgnn_type
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation,True))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation,True))
        # output projection
        # self.gat_layers.append(GATConv(
        #     num_hidden * heads[-2], num_classes, heads[-1],
        #     feat_drop, attn_drop, negative_slope, residual, None))
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_hidden, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None,True))
        self.predict=nn.Linear(num_hidden,num_classes)


    def forward(self, graph,features_list,out):
        keys = list(features_list.keys())
        keys_index = []
        sum = 0
        for key in features_list:
            sum += features_list[key].shape[0]
            keys_index.append(sum)

        feats_emd = [features_list[key] for key in features_list]

        h = []
        for fc, feature in zip(self.fc_list, feats_emd):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for l in range(self.num_layers):
            h = self.gat_layers[l](dgl.to_homogeneous(graph), h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](dgl.to_homogeneous(graph), h).mean(1)
        if(out):
            logits=self.predict(logits)

        res = {}
        for i, key in enumerate(keys):
            if (i == 0):
                res[key] = logits[0:keys_index[i]]
            else:
                res[key] = logits[keys_index[i - 1]:keys_index[i]]
        return res

class HGNN(torch.nn.Module):
    def __init__(self, ntypes ,metadata,hid_dim=None, out_dim=None, num_layer=2, pool=None, hgnn_type='SHGN'
                 , num_heads=8, device=None,dropout=0.2, norm=False,num_etypes=None,input_dims=None,args=None):
        super().__init__()
        self.GraphConv=None
        self.pool=pool
        if hgnn_type == 'HGT':
            self.GraphConv = HGT(node_types=ntypes,metadata=metadata,hidden_channels=hid_dim,out_channels=out_dim,num_heads=num_heads,num_layers=num_layer).to(device)
        elif hgnn_type == 'SHGN':
            heads = [num_heads] * num_layer + [1]
            self.GraphConv = myshgn(args.edge_feats,num_etypes,input_dims,hid_dim,out_dim,num_layer,heads,dropout,ntypes,args.slope)
        elif hgnn_type == 'TransformerConv':
            self.GraphConv = TransformerConv
        else:
            raise KeyError('hgnn_type can be only HGT, SHGN and TransformerConv')

        self.hgnn_type = hgnn_type


        #
        # if pool is None:
        #     self.pool = global_mean_pool
        # else:
        #     self.pool = pool

    def forward(self,targetnode,x,edge_index):
        if self.hgnn_type == 'HGT':
            return self.GraphConv(targetnode,x,edge_index)
        elif self.hgnn_type == 'SHGN':
            g=x
            h_dict=edge_index
            return self.GraphConv(g,h_dict)
        elif self.hgnn_type == 'TransformerConv':
            GraphConv = TransformerConv

        return 11


class HeteroPrompt(torch.nn.Module):
    def __init__(self, token_dims, ntypes):
        super(HeteroPrompt, self).__init__()
        self.token_dims=token_dims
        self.ntypes=ntypes

        self.token_list=torch.nn.ParameterDict()
        for token_dim,node_type in zip(token_dims,ntypes):
            self.token_list[node_type]=torch.nn.Parameter(torch.empty(1,token_dim))

        self.type_token = torch.nn.Parameter(torch.empty(1, len(token_dims)))

        self.token_init(init_method="kaiming_uniform")

    def token_init(self, init_method="kaiming_uniform"):
        if init_method == "kaiming_uniform":
            for key in self.token_list.keys():
                token = self.token_list[key]
                torch.nn.init.kaiming_uniform_(token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
            torch.nn.init.kaiming_uniform_(self.type_token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        else:
            raise ValueError("only support kaiming_uniform init, more init methods will be included soon")


    def forward(self,graph_batch):
        batched_graph=[]

        for graph in dgl.unbatch(graph_batch):
            # device = torch.device("cuda")
            # self.token_list=self.token_list.to(device)
            # graph=graph.to(device=device)
            for node_type in self.token_list.keys():
                newdata=graph.ndata['x'][node_type]*self.token_list[node_type]
                graph.ndata['x']={node_type:newdata}
            batched_graph.append(graph)

        batched_graph=dgl.batch(batched_graph)

        return batched_graph
    def weighted_sum(self,type_embedding):
        token=self.type_token
        token=F.normalize(token,p=2,dim=1)
        graph_embedding=0
        for i,emd in enumerate(type_embedding):
            graph_embedding+=emd*(token[0,i]+1)

        return graph_embedding

class LightPrompt(torch.nn.Module):
    def __init__(self, token_dim, token_num_per_group, group_num=1, inner_prune=None):
        super(LightPrompt, self).__init__()

        self.inner_prune = inner_prune

        self.token_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(token_num_per_group, token_dim)) for i in range(group_num)])

        self.token_init(init_method="kaiming_uniform")

    def token_init(self, init_method="kaiming_uniform"):
        if init_method == "kaiming_uniform":
            for token in self.token_list:
                torch.nn.init.kaiming_uniform_(token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        else:
            raise ValueError("only support kaiming_uniform init, more init methods will be included soon")

    def inner_structure_update(self):
        return self.token_view()

    def token_view(self, ):
        """
        each token group is viewed as a prompt sub-graph.
        turn the all groups of tokens as a batch of prompt graphs.
        :return:
        """
        pg_list = []
        for i, tokens in enumerate(self.token_list):
            # inner link: token-->token
            token_dot = torch.mm(tokens, torch.transpose(tokens, 0, 1))
            token_sim = torch.sigmoid(token_dot)  # 0-1

            inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
            edge_index = inner_adj.nonzero().t().contiguous()

            pg_list.append(Data(x=tokens, edge_index=edge_index, y=torch.tensor([i]).long()))

        pg_batch = Batch.from_data_list(pg_list)
        return pg_batch


class HeavyPrompt(LightPrompt):
    def __init__(self, token_dim, token_num, cross_prune=0.1, inner_prune=0.01):
        super(HeavyPrompt, self).__init__(token_dim, token_num, 1, inner_prune)  # only has one prompt graph.
        self.cross_prune = cross_prune

    def forward(self, graph_batch: Batch):
        """
        TODO: although it recieves graph batch, currently we only implement one-by-one computing instead of batch computing
        TODO: we will implement batch computing once we figure out the memory sharing mechanism within PyG
        :param graph_batch:
        :return:
        """
        # device = torch.device("cuda")
        # device = torch.device("cpu")

        pg = self.inner_structure_update()  # batch of prompt graph (currently only 1 prompt graph in the batch)

        inner_edge_index = pg.edge_index
        token_num = pg.x.shape[0]

        re_graph_list = []
        for g in Batch.to_data_list(graph_batch):
            g_edge_index = g.edge_index + token_num
            # pg_x = pg.x.to(device)
            # g_x = g.x.to(device)

            cross_dot = torch.mm(pg.x, torch.transpose(g.x, 0, 1))
            cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
            cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)

            cross_edge_index = cross_adj.nonzero().t().contiguous()
            cross_edge_index[1] = cross_edge_index[1] + token_num

            x = torch.cat([pg.x, g.x], dim=0)
            y = g.y

            edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)
            data = Data(x=x, edge_index=edge_index, y=y)
            re_graph_list.append(data)

        graphp_batch = Batch.from_data_list(re_graph_list)
        return graphp_batch


class FrontAndHead(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=16, num_classes=2,
                 task_type="multi_label_classification",
                 token_num=10, cross_prune=0.1, inner_prune=0.3):

        super().__init__()

        self.PG = HeavyPrompt(token_dim=input_dim, token_num=token_num, cross_prune=cross_prune,
                              inner_prune=inner_prune)

        if task_type == 'multi_label_classification':
            self.answering = torch.nn.Sequential(
                torch.nn.Linear(hid_dim, num_classes),
                torch.nn.Softmax(dim=1))
        else:
            raise NotImplementedError

    def forward(self, graph_batch, gnn):
        prompted_graph = self.PG(graph_batch)
        graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
        pre = self.answering(graph_emb)

        return pre





if __name__ == '__main__':
    pass

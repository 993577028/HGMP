import os
import numpy as np
import random
from copy import deepcopy
from random import shuffle
import torch_geometric.transforms as T
from scipy.sparse import csr_matrix
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import subgraph, k_hop_subgraph,to_dgl,to_networkx
from torch_geometric.loader import HGTLoader,ClusterData
from torch_geometric.datasets import HGBDataset,IMDB
from torch_geometric.transforms import ToUndirected
from .data_loader import data_loader
import torch_geometric
from torch_geometric.data import HeteroData
import dgl
import scipy.sparse as sp

import torch.nn.functional as F
from scripts.constant import data_target
import tqdm

import copy
import os.path as osp
import sys
from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
import torch.utils.data
from torch import Tensor

import torch_geometric.typing
from torch_geometric.data import Data
from torch_geometric.typing import pyg_lib
from torch_geometric.utils import index_sort, narrow, select, sort_edge_index,remove_isolated_nodes
from torch_geometric.utils.map import map_index
from torch_geometric.utils.sparse import index2ptr, ptr2index


seed = 0


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("create folder {}".format(path))
    else:
        print("folder exists! {}".format(path))


# used in pre_train.py
def gen_ran_output(targetnode,data, model):
    vice_model = deepcopy(model)

    for (vice_name, vice_model_param), (name, param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if vice_name.split('.')[0] == 'projection_head':
            vice_model_param.data = param.data
        else:
            vice_model_param.data = param.data + 0.1 * torch.normal(0, torch.ones_like(
                param.data) * param.data.std())

    z2 = vice_model.forward_cl_gcn(targetnode, data)

    return z2

def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)



def dense_to_sparse(dense_tensor):
    """
    将密集张量转换为稀疏张量。

    参数:
    dense_tensor (torch.Tensor): 输入的密集张量。

    返回:
    torch.sparse_coo_tensor: 输出的稀疏张量。
    """
    # 找出非零元素的索引和值
    indices = dense_tensor.nonzero(as_tuple=False)
    indices = indices.t().contiguous()
    values = dense_tensor[dense_tensor != 0]

    # 创建稀疏张量
    sparse_tensor = torch.sparse_coo_tensor(indices, values, size=dense_tensor.size())

    return sparse_tensor
# used in pre_train.py

def load_data4pretrain_metapath(feats_type=3,device=None,dataname='IMDB',batch_size=32,num_sample=128):
    if(dataname=='Freebase' and feats_type==-1):
        dataset = HGBDataset(root='./dataset', name=dataname,transform=ToUndirected(merge=False))
    else:
        dataset = HGBDataset(root='./dataset', name=dataname)

    if dataname == 'IMDB':
        target = 'movie'
        num_class = 5
        shots = 50
        metapath = [
            [('movie', 'actor'), ('actor', 'movie')],  # MAM
            [('movie', 'director'), ('director', 'movie')],  # MDM
        ]
    elif dataname == 'ACM':
        num_class = 3
        target = 'paper'
        shots = 30
        metapath = [[('paper', 'author'), ('author', 'paper')],  # PAP
                    [('paper', 'subject'), ('subject', 'paper')]  # PSP
                    ]
    elif dataname == 'DBLP':
        num_class = 4
        shots = 40
        target = 'author'
        metapath = [[('author', 'paper'), ('paper', 'author')]  # APA
                    # [('author', 'paper'), ('paper', 'term'), ('term', 'paper'), ('paper', 'author')],  # APTPA
                    # [('author', 'paper'), ('paper', 'venue'), ('venue', 'paper'), ('paper', 'author')]  # APVPA
                    ]
    data = T.AddMetaPaths(metapath)(dataset[0])


    for node_type, node_store in data.node_items():
        for attr, value in node_store.items():
            #没有节点属性的节点类型赋值对角矩阵
            if attr=='num_nodes':
                data[node_type]['x']=torch.eye(value)
                del data[node_type][attr]


    features=data.x_dict
    features_list=[]
    for value in features.values():
        features_list.append(value)

    if feats_type == 0 or feats_type == -1:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []  # [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10))
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim]))
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim]))

    value_dict={}
    i = 0
    for ntype in data.node_types:
        value_dict[ntype]=features_list[i]
        i=i+1

    data.set_value_dict('x',value_dict)

    num_class=(torch.max(data[data_target[dataname]].y)+1).item()


    node_type=data.node_types
    name4dim={}
    for name,dim in zip(node_type,in_dims):
        name4dim[name]=dim

    pyg_graph_list = ClusterData(data=data.to_homogeneous(), num_parts=num_sample)

    pyg_graph_list=list(pyg_graph_list)

    graph_list=[]
    graph_list1=[]
    for i,graph in enumerate(pyg_graph_list):
        graph=graph.to_heterogeneous()
        node_types=graph.node_types
        for node_type in node_types:
            # 检查当前类型的节点是否有属性'x'
            if 'x' in graph[node_type]:
                value=graph[node_type]['x'].to_dense()
                value=value[: ,:name4dim[node_type]]
                # value=dense_to_sparse(value)
                graph[node_type]['x']=value


        graph_list1.append(pyg_to_dgl(graph))
        #graph_list.append(graph)

    # pyg转dgl后节点类型顺序会被打乱
    ACM_indices = [1, 0, 2, 3]
    IMDB_indices = [2, 1, 3, 0]
    Freebase_indices=[0,7,1,5,2,6,4,3]
    if(dataname=='ACM'):
        in_dims = [in_dims[i] for i in ACM_indices]
    elif(dataname=='IMDB'):
        in_dims = [in_dims[i] for i in IMDB_indices]
    elif(dataname=='Freebase'):
        in_dims = [in_dims[i] for i in Freebase_indices]

    return graph_list1,in_dims,num_class
def load_data4pretrain(feats_type=3,device=None,dataname='IMDB',batch_size=32,num_sample=128):
    if(dataname=='Freebase' and feats_type==-1):
        dataset = HGBDataset(root='./dataset', name=dataname,transform=ToUndirected(merge=False))
        data = dataset[0]
    elif(dataname=='oldfreebase'):
        data=load_freebase("./dataset/oldfreebase")
    else:
        dataset = HGBDataset(root='./dataset', name=dataname)
        data=dataset[0]


    for node_type, node_store in data.node_items():
        for attr, value in node_store.items():
            #没有节点属性的节点类型赋值对角矩阵
            if attr=='num_nodes':
                #data[node_type]['x']=torch.eye(value)
                data[node_type]['x'] = create_matrix(value, 0.01)
                del data[node_type][attr]


    features=data.x_dict
    features_list=[]
    for value in features.values():
        features_list.append(value)

    if feats_type == 0 or feats_type == -1:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []  # [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10))
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim]))
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim]))

    value_dict={}
    i = 0
    for ntype in data.node_types:
        value_dict[ntype]=features_list[i]
        i=i+1

    data.set_value_dict('x',value_dict)
    num_class=(torch.max(data[data_target[dataname]].y)+1).item()

    # graph_loader = HGTLoader(
    #     data,
    #     # Sample 512 nodes per type and per iteration for 4 iterations
    #     num_samples={key: [num_sample] * 2 for key in data.node_types},
    #     # Use a batch size of 128 for sampling training nodes of type target
    #     batch_size=batch_size,
    #     input_nodes=data_target[dataname]
    # )
    #
    # pyg_graph_list=list(graph_loader)

    node_type=data.node_types
    name4dim={}
    for name,dim in zip(node_type,in_dims):
        name4dim[name]=dim

    pyg_graph_list = ClusterData(data=data.to_homogeneous(), num_parts=num_sample)

    pyg_graph_list=list(pyg_graph_list)

    graph_list=[]
    graph_list1=[]
    for i,graph in enumerate(pyg_graph_list):
        graph=graph.to_heterogeneous()
        node_types=graph.node_types
        for node_type in node_types:
            # 检查当前类型的节点是否有属性'x'
            if 'x' in graph[node_type]:
                value=graph[node_type]['x'].to_dense()
                value=value[: ,:name4dim[node_type]]
                # value=dense_to_sparse(value)
                graph[node_type]['x']=value


        graph_list1.append(pyg_to_dgl(graph))
        #graph_list.append(graph)

    # pyg转dgl后节点类型顺序会被打乱
    ACM_indices = [1, 0, 2, 3]
    IMDB_indices = [2, 1, 3, 0]
    Freebase_indices=[0,7,1,5,2,6,4,3]
    oldfreebase_indices=[1,2,0,3]
    if(dataname=='ACM'):
        in_dims = [in_dims[i] for i in ACM_indices]
    elif(dataname=='IMDB'):
        in_dims = [in_dims[i] for i in IMDB_indices]
    elif(dataname=='Freebase'):
        in_dims = [in_dims[i] for i in Freebase_indices]
    elif (dataname == 'oldfreebase'):
        in_dims = [in_dims[i] for i in oldfreebase_indices]

    return graph_list1,in_dims,num_class

def new_load_data4pretrain(feats_type=3,device=None,dataname='IMDB',batch_size=32,num_sample=128):
    if(dataname=='Freebase' and feats_type==-1):
        dataset = HGBDataset(root='./dataset', name=dataname,transform=ToUndirected(merge=False))
    else:
        dataset = HGBDataset(root='./dataset', name=dataname)
    data=dataset[0]

    if dataname=='IMDB':
        targetnode='movie'
    elif dataname=='DBLP':
        targetnode = 'author'
    elif dataname=='ACM':
        targetnode = 'paper'

    for node_type, node_store in data.node_items():
        for attr, value in node_store.items():
            #没有节点属性的节点类型赋值对角矩阵
            if attr=='num_nodes':
                data[node_type]['x']=torch.eye(value)
                del data[node_type][attr]


    features=data.x_dict
    features_list=[]
    for value in features.values():
        features_list.append(value)

    if feats_type == 0 or feats_type == -1:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []  # [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10))
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim]))
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim]))

    value_dict={}
    i = 0
    for ntype in data.node_types:
        value_dict[ntype]=features_list[i]
        i=i+1

    data.set_value_dict('x',value_dict)

    num_class=(torch.max(data[data_target[dataname]].y)+1).item()

    #用元路径来划分partition
    adjs=new_loaddata(dataname)
    sparse_matrix = csr_matrix(adjs)
    row, col = sparse_matrix.nonzero()
    edge_index = torch.tensor([row, col], dtype=torch.long)  # 边的索引
    edge_weight = torch.tensor(sparse_matrix.data, dtype=torch.float)  # 边的权重

    # 创建PyG的Data对象
    mg_data = Data(edge_index=edge_index, edge_attr=edge_weight,num_nodes=features_list[0].shape[0])
    pyg_graph_list = ClusterData(data=mg_data, num_parts=num_sample)
    partition=pyg_graph_list.partition
    index_list=[]
    for i in range(num_sample):
        start=partition.partptr[i].item()
        end=partition.partptr[i+1].item()
        index_list.append(partition.node_perm[start:end])

    graph_list=[]
    data=to_dgl(data)

    for index in index_list:
        graph,_=dgl.khop_in_subgraph(data, {targetnode: index}, k=1)
        graph_list.append(graph)




    # pyg转dgl后节点类型顺序会被打乱
    ACM_indices = [1, 0, 2, 3]
    IMDB_indices = [2, 1, 3, 0]
    Freebase_indices=[0,7,1,5,2,6,4,3]
    if(dataname=='ACM'):
        in_dims = [in_dims[i] for i in ACM_indices]
    elif(dataname=='IMDB'):
        in_dims = [in_dims[i] for i in IMDB_indices]
    elif(dataname=='Freebase'):
        in_dims = [in_dims[i] for i in Freebase_indices]

    return graph_list,in_dims,num_class


def pyg_to_dgl(data):
    data_dict = {}
    for edge_type, edge_store in data.edge_items():
        if edge_store.get('edge_index') is not None:
            row, col = edge_store.edge_index
        else:
            row, col, _ = edge_store['adj_t'].t().coo()

        data_dict[edge_type] = (row, col)
    g = dgl.heterograph(data_dict)

    for node_type, node_store in data.node_items():
        num_nodes=g.num_nodes(node_type)
        for attr, value in node_store.items():
            if(attr=='x'):
                g.nodes[node_type].data[attr] = value[:num_nodes]

    for edge_type, edge_store in data.edge_items():
        for attr, value in edge_store.items():
            if attr in ['edge_index', 'adj_t']:
                continue
            g.edges[edge_type].data[attr] = value

    return g

def edgeindex_to_adjacency_matrix(edge_index, num_nodes):
    # 初始化一个全零的邻接矩阵
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # 获取边的起始节点和终止节点
    rows = edge_index[0].numpy()  # 起始节点
    cols = edge_index[1].numpy()  # 终止节点

    # 将邻接矩阵相应的位置设为 1
    adjacency_matrix[rows, cols] = 1  # 若为无向图，也可以反向设置 adjacency_matrix[cols, rows] = 1

    return adjacency_matrix
def new_loaddata(dataname):
    dataset = HGBDataset(root='./dataset', name=dataname)
    if dataname == 'IMDB':
        target = 'movie'
        num_class=5
        shots=50
        metapath = [
            [('movie', 'actor'), ('actor', 'movie')],  # MAM
            [('movie', 'director'), ('director', 'movie')],  # MDM
        ]
    elif dataname == 'ACM':
        num_class = 3
        target = 'paper'
        shots=30
        metapath = [[('paper', 'author'), ('author', 'paper')],  # PAP
                    [('paper', 'subject'), ('subject', 'paper')]  # PSP
                    ]
    elif dataname == 'DBLP':
        num_class = 4
        shots = 40
        target = 'author'
        metapath = [[('author', 'paper'), ('paper', 'author')],  # APA
                    [('author', 'paper'), ('paper', 'term'), ('term', 'paper'), ('paper', 'author')],  # APTPA
                    [('author', 'paper'), ('paper', 'venue'), ('venue', 'paper'), ('paper', 'author')]  # APVPA
                    ]
    data = T.AddMetaPaths(metapath, drop_orig_edge_types=True)(dataset[0])
    res={}
    num_nodes=data[target].x.shape[0]
    label=data[target].y.to_dense()
    if(dataname!='IMDB'):
        label = torch.nn.functional.one_hot(label, num_classes=num_class)
    res['label']=label.numpy()

    res['feature']=data[target].x.to_dense().numpy()
    res['train_idx'] = np.where(data[target].train_mask.to_dense().numpy())[0].reshape(1,-1)[:,:shots]
    res['test_idx'] = np.where(data[target].test_mask.to_dense().numpy())[0].reshape(1,-1)

    if(dataname=='ACM'):
        for edge_type, edge_store in data.edge_items():
            if (edge_type == ('paper', 'metapath_0', 'paper')):
                pap = edge_store['edge_index']
            elif (edge_type == ('paper', 'metapath_1', 'paper')):
                psp = edge_store['edge_index']
        res['pap']=edgeindex_to_adjacency_matrix(pap,num_nodes)
        res['psp']=edgeindex_to_adjacency_matrix(psp,num_nodes)
        adjs=res['pap']+res['psp']
    elif(dataname=='DBLP'):
        for edge_type, edge_store in data.edge_items():
            if (edge_type == ('author', 'metapath_0', 'author')):
                apa = edge_store['edge_index']
            elif (edge_type == ('author', 'metapath_1', 'author')):
                aptpa = edge_store['edge_index']
            elif (edge_type == ('author', 'metapath_2', 'author')):
                apvpa = edge_store['edge_index']
        res['apa']=edgeindex_to_adjacency_matrix(apa,num_nodes)
        res['aptpa'] = edgeindex_to_adjacency_matrix(aptpa,num_nodes)
        res['apvpa'] = edgeindex_to_adjacency_matrix(apvpa,num_nodes)
        adjs=res['apa']+res['aptpa']+res['apvpa']
    elif(dataname=='IMDB'):
        for edge_type, edge_store in data.edge_items():
            if (edge_type == ('movie', 'metapath_0', 'movie')):
                mam = edge_store['edge_index']
            elif (edge_type == ('movie', 'metapath_1', 'movie')):
                mdm = edge_store['edge_index']
        res['mam']=edgeindex_to_adjacency_matrix(mam,num_nodes)
        res['mdm']=edgeindex_to_adjacency_matrix(mdm,num_nodes)
        adjs=res['mam']+res['mdm']
    return adjs



# used in prompt.py
def act(x=None, act_type='leakyrelu'):
    if act_type == 'leakyrelu':
        if x is None:
            return torch.nn.LeakyReLU()
        else:
            return F.leaky_relu(x)
    elif act_type == 'tanh':
        if x is None:
            return torch.nn.Tanh()
        else:
            return torch.tanh(x)

def load_data(prefix='DBLP'):
    dl = data_loader('./dataset/'+prefix)
    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)
    adjM = sum(dl.links['data'].values())
    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    val_ratio = 0.2
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[:split]
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    labels[test_idx] = dl.labels_test['data'][dl.labels_test['mask']]
    if prefix != 'IMDB':
        labels = labels.argmax(axis=1)
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = train_idx
    train_val_test_idx['val_idx'] = val_idx
    train_val_test_idx['test_idx'] = test_idx
    return features,\
           adjM, \
           labels,\
           train_val_test_idx,\
            dl


def __seeds_list__(nodes):
    split_size = max(5, int(nodes.shape[0] / 400))
    seeds_list = list(torch.split(nodes, split_size))
    if len(seeds_list) < 400:
        print('len(seeds_list): {} <400, start overlapped split'.format(len(seeds_list)))
        seeds_list = []
        while len(seeds_list) < 400:
            split_size = random.randint(3, 5)
            seeds_list_1 = torch.split(nodes, split_size)
            seeds_list = seeds_list + list(seeds_list_1)
            nodes = nodes[torch.randperm(nodes.shape[0])]
    shuffle(seeds_list)
    seeds_list = seeds_list[0:400]

    return seeds_list


def __dname__(p, task_id):
    if p == 0:
        dname = 'task{}.meta.train.support'.format(task_id)
    elif p == 1:
        dname = 'task{}.meta.train.query'.format(task_id)
    elif p == 2:
        dname = 'task{}.meta.test.support'.format(task_id)
    elif p == 3:
        dname = 'task{}.meta.test.query'.format(task_id)
    else:
        raise KeyError

    return dname


def __pos_neg_nodes__(labeled_nodes, node_labels, i: int):
    pos_nodes = labeled_nodes[node_labels[:, i] == 1]
    pos_nodes = pos_nodes[torch.randperm(pos_nodes.shape[0])]
    neg_nodes = labeled_nodes[node_labels[:, i] == 0]
    neg_nodes = neg_nodes[torch.randperm(neg_nodes.shape[0])]
    return pos_nodes, neg_nodes


def __induced_graph_list_for_graphs__(seeds_list, label, p, num_nodes, potential_nodes, ori_x, same_label_edge_index,
                                      smallest_size, largest_size):
    seeds_part_list = seeds_list[p * 100:(p + 1) * 100]
    induced_graph_list = []
    for seeds in seeds_part_list:

        subset, _, _, _ = k_hop_subgraph(node_idx=torch.flatten(seeds), num_hops=1, num_nodes=num_nodes,
                                         edge_index=same_label_edge_index, relabel_nodes=True)

        temp_hop = 1
        while len(subset) < smallest_size and temp_hop < 5:
            temp_hop = temp_hop + 1
            subset, _, _, _ = k_hop_subgraph(node_idx=torch.flatten(seeds), num_hops=temp_hop, num_nodes=num_nodes,
                                             edge_index=same_label_edge_index, relabel_nodes=True)

        if len(subset) < smallest_size:
            need_node_num = smallest_size - len(subset)
            candidate_nodes = torch.from_numpy(np.setdiff1d(potential_nodes.numpy(), subset.numpy()))

            candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]

            subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

        if len(subset) > largest_size:
            # directly downmsample
            subset = subset[torch.randperm(subset.shape[0])][0:largest_size - len(seeds)]
            subset = torch.unique(torch.cat([torch.flatten(seeds), subset]))

        sub_edge_index, _ = subgraph(subset, same_label_edge_index, num_nodes=num_nodes, relabel_nodes=True)

        x = ori_x[subset]
        graph = Data(x=x, edge_index=sub_edge_index, y=label)
        induced_graph_list.append(graph)

    return induced_graph_list


def graph_views(data, aug='random', aug_ratio=0.1):
    if aug == 'dropN':
        data = drop_nodes(data, aug_ratio)
    elif aug == 'permE':
        data = permute_edges(data, aug_ratio,1.5)
    elif aug == 'maskN':
        data = mask_nodes(data, aug_ratio,1.5)
    elif aug == 'randomDropN':
        data = random_remove_nodes(data, aug_ratio)
    elif aug == 'randomPermE':
        data = random_permute_edges(data, aug_ratio)
    elif aug == 'random':
        n = np.random.randint(2)
        if n == 0:
            data = drop_nodes(data, aug_ratio)
        elif n == 1:
            data = permute_edges(data, aug_ratio)
        else:
            print('augmentation error')
            assert False
    return data


def random_remove_nodes(data, aug_ratio):
    """
    根据 aug_ratio 在全局范围内随机删除节点，不考虑节点类型。

    Parameters:
    - data (DGLHeteroGraph): 输入的异构图。
    - aug_ratio (float): 节点删除比例，范围为 [0, 1]。

    Returns:
    - DGLHeteroGraph: 删除部分节点后的异构图。
    """
    if not (0 <= aug_ratio <= 1):
        raise ValueError("aug_ratio must be between 0 and 1.")

    # Step 1: 计算总节点数
    node_types = data.ntypes
    total_nodes = sum(data.num_nodes(ntype) for ntype in node_types)

    # Step 2: 计算需要删除的节点数
    num_nodes_to_remove = int(total_nodes * aug_ratio)
    if num_nodes_to_remove == 0:
        return data  # 如果 aug_ratio 太小，返回原图

    # Step 3: 随机选择节点进行删除
    all_nodes = torch.arange(total_nodes)
    nodes_to_remove = torch.randperm(total_nodes)[:num_nodes_to_remove]

    # Step 4: 删除节点
    nodes_removed = 0
    for ntype in node_types:
        num_nodes = data.num_nodes(ntype)
        nodes_of_type = nodes_to_remove[
                            (nodes_to_remove >= nodes_removed) & (nodes_to_remove < nodes_removed + num_nodes)
                            ] - nodes_removed
        if len(nodes_of_type) > 0:
            data = dgl.remove_nodes(data, nodes_of_type, ntype=ntype)
        nodes_removed += num_nodes

    return data

def random_permute_edges(data, aug_ratio):
    """
    随机删除异构图中的边，不区分边类型。

    Parameters:
    - data: DGLHeteroGraph，输入的异构图。
    - aug_ratio: float，随机删除边的比例，范围在 [0, 1]。

    Returns:
    - DGLHeteroGraph: 删除部分边后的异构图。
    """
    if not (0 <= aug_ratio <= 1):
        raise ValueError("aug_ratio must be between 0 and 1.")

    # 获取所有边类型的总数
    all_edge_types = data.canonical_etypes
    all_edges = []
    edge_offsets = []
    total_edges = 0

    # 收集每种类型的边，并计算偏移
    for etype in all_edge_types:
        num_edges = data.num_edges(etype)
        total_edges += num_edges
        edge_offsets.append(total_edges)
        all_edges.append((etype, torch.arange(num_edges)))

    if total_edges == 0:
        raise ValueError("The graph has no edges to remove.")

    # 计算需要删除的边数
    num_edges_to_remove = int(total_edges * aug_ratio)
    if num_edges_to_remove == 0:
        return data

    # 将所有边打平，统一编号
    flat_edges = torch.cat([edges for _, edges in all_edges])
    flat_edge_types = torch.cat([torch.full_like(edges, idx, dtype=torch.int64)
                                 for idx, (etype, edges) in enumerate(all_edges)])

    # 随机选择边的索引进行删除
    indices_to_remove = torch.randperm(total_edges)[:num_edges_to_remove]
    edges_to_remove = flat_edges[indices_to_remove]
    types_to_remove = flat_edge_types[indices_to_remove]

    # 删除边
    for etype_idx, etype in enumerate(all_edge_types):
        mask = types_to_remove == etype_idx
        if mask.sum() > 0:
            edges_to_delete = edges_to_remove[mask]
            data = dgl.remove_edges(data, edges_to_delete, etype=etype)

    return data

def drop_nodes(data, aug_ratio):
    num_nodes=data.num_nodes()
    saved_num=int(num_nodes*(1-aug_ratio))
    proportional_sizes = calculate_proportional_size(data, saved_num)
    sampled_nodes = sample_nodes(data, proportional_sizes, {})

    new_data=subgraph_from_nodes(data,sampled_nodes)

    return new_data


def permute_edges(hetero_graph, ratio,power):
    """
    计算每种边的比例，然后用 softmax 平滑后按照新的比例删除边。

    Parameters:
    - hetero_graph: DGLHeteroGraph，异构图对象
    - ratio: float，表示基础删除比例

    Returns:
    - DGLHeteroGraph: 删除部分边后的异构图
    """
    if not (0 <= ratio <= 1):
        raise ValueError("Ratio must be between 0 and 1.")

    edge_types = hetero_graph.canonical_etypes
    num_edges_list = [hetero_graph.num_edges(etype) for etype in edge_types]
    total_edges = sum(num_edges_list)
    # Step 1: 计算每种边的比例
    edge_ratios = torch.tensor([num_edges / total_edges for num_edges in num_edges_list], dtype=torch.float32)

    adjusted_ratios = edge_ratios ** power
    normalized_ratios = (adjusted_ratios / adjusted_ratios.sum()) * ratio

    edges_to_remove = {etype: [] for etype in edge_types}

    # Step 2: 按新的比例删除边
    for etype, adjusted_ratio in zip(edge_types, normalized_ratios):
        num_edges = hetero_graph.num_edges(etype)
        num_edges_to_remove = int(total_edges * adjusted_ratio.item())  # 删除的边数量

        if num_edges_to_remove > 0:
            all_edges = torch.arange(num_edges)
            edges_to_remove[etype] = torch.randperm(num_edges)[:num_edges_to_remove]

    for etype, edges in edges_to_remove.items():
        if len(edges) > 0:
            hetero_graph = dgl.remove_edges(hetero_graph, edges, etype=etype)

    return hetero_graph

def mask_nodes(hetero_graph, ratio,power):
    """
    根据节点类型的比例，应用 softmax 后随机替换节点特征。

    Parameters:
    - hetero_graph: DGLHeteroGraph，异构图对象
    - ratio: float，表示基础替换比例

    Returns:
    - DGLHeteroGraph: 替换部分节点特征后的异构图
    """
    if not (0 <= ratio <= 1):
        raise ValueError("Ratio must be between 0 and 1.")

    node_types = hetero_graph.ntypes
    num_nodes_list = [hetero_graph.num_nodes(ntype) for ntype in node_types]
    total_nodes = sum(num_nodes_list)

    # Step 1: 计算每种节点类型的比例，并应用 softmax
    node_ratios = torch.tensor([num_nodes / total_nodes for num_nodes in num_nodes_list], dtype=torch.float32)

    adjusted_ratios = node_ratios ** power
    normalized_ratios = (adjusted_ratios / adjusted_ratios.sum()) * ratio

    # Step 2: 按调整后的比例替换节点特征
    for ntype, adjusted_ratio in zip(node_types, normalized_ratios):
        num_nodes = hetero_graph.num_nodes(ntype)
        num_nodes_to_replace = int(total_nodes * adjusted_ratio.item())

        if num_nodes_to_replace > 0:
            # 计算该类型节点的特征均值
            node_features = hetero_graph.nodes[ntype].data['x']  # 假设特征名称为 'feat'
            mean_embedding = node_features.mean(dim=0, keepdim=True)

            # 随机选择一些节点进行特征替换
            all_nodes = torch.arange(num_nodes)
            nodes_to_replace = torch.randperm(num_nodes)[:num_nodes_to_replace]

            # 替换特征
            hetero_graph.nodes[ntype].data['x'][nodes_to_replace] = mean_embedding

    return hetero_graph

def delete_edges_by_ratio(data: HeteroData, ratio: float) -> HeteroData:
    # 复制数据，以防止修改原始数据
    data = data.clone()

    # 遍历每种类型的边
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        num_edges = edge_index.size(1)
        delete_count = int(num_edges * ratio)

        # 获取所有边的索引
        edge_indices = list(range(num_edges))

        # 随机选择需要删除的边
        delete_indices = set(random.sample(edge_indices, delete_count))

        # 创建保留边的掩码
        mask = torch.tensor([i not in delete_indices for i in edge_indices], dtype=torch.bool)

        # 更新边索引
        data[edge_type].edge_index = edge_index[:, mask]

    return data


def delete_nodes_by_ratio_using_subgraph(data: HeteroData, ratio: float) -> HeteroData:
    # 复制数据，以防止修改原始数据
    data = data.clone()

    # 用于存储需要保留节点的索引
    keep_node_indices = {}

    # 计算每种类型节点需要删除的数量，并选择需要保留的节点
    for node_type in data.node_types:
        node_count = data[node_type].num_nodes
        delete_count = int(node_count * ratio)

        # 获取所有节点的索引
        node_indices = list(range(node_count))

        # 随机选择需要保留的节点
        keep_count = node_count - delete_count
        keep_indices = random.sample(node_indices, keep_count)

        # 记录需要保留的节点索引
        keep_node_indices[node_type] = torch.tensor(keep_indices, dtype=torch.long)

    # 使用subgraph函数创建子图
    subgraph_data = data.subgraph(keep_node_indices)

    return subgraph_data

def sample_nodes(g, proportional_sizes, specified_nodes):
    sampled_nodes = {}
    for ntype, size in proportional_sizes.items():
        total_nodes = g.num_nodes(ntype)
        specified_nodes_ntype = specified_nodes.get(ntype, [])
        if(type(specified_nodes_ntype)!=list):
            specified_nodes_ntype=specified_nodes_ntype.tolist()
        num_specified = len(specified_nodes_ntype)

        if size >= total_nodes:
            sampled_nodes[ntype] = torch.arange(total_nodes)
        else:
            if num_specified > size:
                raise ValueError(f"Too many specified nodes for node type {ntype}.")
            remaining_size = size - num_specified
            remaining_nodes = list(set(range(total_nodes)) - set(specified_nodes_ntype))
            if remaining_size > 0:
                sampled_remaining_nodes = np.random.choice(remaining_nodes, remaining_size, replace=False)
            else:
                sampled_remaining_nodes = []
            sampled_nodes[ntype] = torch.tensor(specified_nodes_ntype + list(sampled_remaining_nodes))
    return sampled_nodes

def calculate_proportional_size(g, target_total_size):
    node_counts = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
    total_nodes = sum(node_counts.values())
    proportional_sizes = {ntype: int(target_total_size * (count / total_nodes)) for ntype, count in node_counts.items()}

    # Ensure at least one node is selected for each type if possible
    for ntype in g.ntypes:
        if proportional_sizes[ntype] == 0 and node_counts[ntype] > 0:
            proportional_sizes[ntype] = 1

    return proportional_sizes

def subgraph_from_nodes(g, sampled_nodes):
    subgraph = dgl.node_subgraph(g, sampled_nodes)
    return subgraph


def graph_pool(pool,node_emb,prompted_graph):
    type_list = []
    for node_type, x in node_emb.items():
        if (prompted_graph.num_nodes(node_type) == 0):
            continue
        seg = prompted_graph.batch_num_nodes(node_type)
        val = x
        type_list.append(dgl.ops.segment_reduce(seg, val, 'mean'))

    graph_emb = torch.stack(type_list)
    if(type(pool)!=str):
        graph_emb=pool.weighted_sum(graph_emb)
    elif(pool=='sum'):
        graph_emb = graph_emb.sum(dim=0)
    elif(pool=='mean'):
        graph_emb = graph_emb.mean(dim=0)
    elif (pool == 'max'):
        graph_emb = graph_emb.max(dim=0)

    return graph_emb

def load_freebase(path):
    ma = np.genfromtxt(path + "/ma.txt",dtype=np.int64)
    md = np.genfromtxt(path +"/md.txt",dtype=np.int64)
    mw = np.genfromtxt(path +"/mw.txt",dtype=np.int64)
    label = np.load(path +"/labels.npy").astype('int64')
    mp2vec=torch.load(path +"/mp2vec.pth")

    ma_src, ma_dst = zip(*ma)
    md_src, md_dst = zip(*md)
    mw_src, mw_dst = zip(*mw)


    # 创建异构图
    data = HeteroData()
    #data['movie'].num_nodes = 3492
    # 添加节点标签
    data['movie'].y=torch.tensor(label, dtype=torch.int64)
    data['movie'].x = mp2vec.to('cpu')

    data['actor'].num_nodes = 33401
    data['director'].num_nodes = 2502
    data['writer'].num_nodes = 4459

    # 创建正向边
    data['movie', 'ma', 'actor'].edge_index = torch.tensor([ma_src, ma_dst], dtype=torch.int64)
    data['movie', 'md', 'director'].edge_index = torch.tensor([md_src, md_dst], dtype=torch.int64)
    data['movie', 'mw', 'writer'].edge_index = torch.tensor([mw_src, mw_dst], dtype=torch.int64)

    # 创建反向边
    data['actor', 'am', 'movie'].edge_index = torch.tensor([ma_dst, ma_src], dtype=torch.int64)
    data['director', 'dm', 'movie'].edge_index = torch.tensor([md_dst, md_src], dtype=torch.int64)
    data['writer', 'wm', 'movie'].edge_index = torch.tensor([mw_dst, mw_src], dtype=torch.int64)


    return data



def set_params(args):
    if(args.dataset=='ACM'):
        args.hidden_dim=512
        args.num_samples=500
        args.prompt_lr=5e-2
        args.head_lr=5e-4
        args.weight_decay=5e-5
        args.pre_lr = 1e-4
        args.aug_ration = 0.1
    elif(args.dataset == 'IMDB'):
        args.hidden_dim = 256
        args.num_samples = 200
        args.prompt_lr = 5e-2
        args.head_lr = 5e-2
        args.weight_decay = 1e-4
        args.pre_lr = 1e-3
        args.aug_ration = 0.3 #0.2
    elif(args.dataset == 'oldfreebase'):
        args.hidden_dim = 256
        args.num_samples = 450
        args.prompt_lr = 5e-3
        args.head_lr = 5e-3
        args.weight_decay = 1e-5
        args.pre_lr = 5e-3
        args.aug_ration = 0.1

    return args



def create_matrix(size, off_diag_value=0.1):
    """
    创建一个对角线元素为 1，其他元素为 off_diag_value 的矩阵。

    参数:
    size (int): 矩阵的大小 (size x size)。
    off_diag_value (float): 非对角线元素的值，默认为 0.1。

    返回:
    torch.Tensor: 生成的矩阵。
    """
    # 创建一个全为 off_diag_value 的 size x size 矩阵
    matrix = torch.full((size, size), off_diag_value)

    # 将对角线元素替换为 1
    for i in range(size):
        matrix[i, i] = 1.0

    return matrix




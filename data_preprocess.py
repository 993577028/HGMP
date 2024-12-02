from collections import defaultdict
import pickle as pk
from torch_geometric.utils import to_dgl
import torch
import os
from torch_geometric.datasets import HGBDataset
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import ToUndirected
import random
import warnings
from HGP.utils import mkdir,load_freebase,create_matrix
from random import shuffle
from torch_geometric.data import HeteroData

os.environ['HTTP_PROXY'] = "http://10.14.117.235:7890"
os.environ['HTTPS_PROXY'] = "http://10.14.117.235:7890"


import dgl
import numpy as np

def nodes_split(data: Data, dataname: str = None, node_classes=3, targetnode=None):
    if dataname is None:
        raise KeyError("dataname is None!")

    index_path = './dataset/{}/index/'.format(dataname.lower())
    mkdir(index_path)

    node_labels = data.ndata['y'][targetnode]

    # step1: split/sample nodes
    for i in range(0, node_classes):
        pos_nodes = torch.argwhere(node_labels == i)  # torch.squeeze(torch.argwhere(node_labels == i))
        pos_nodes = pos_nodes[torch.randperm(pos_nodes.shape[0])]
        # TODO: ensure each label contain more than 400 nodes

        if pos_nodes.shape[0] < 400:
            warnings.warn("label {} only has {} nodes but it should be larger than 400!".format(i, pos_nodes.shape[0]),
                          RuntimeWarning)
        else:
            pos_nodes = pos_nodes[0:400]
        # print(pos_nodes.shape)


        dname = 'task{}'.format(i)
        pk.dump(pos_nodes, open(index_path + dname, 'bw'))


def edge_split(data, dataname: str = None, node_classes=3, targetnode=None):
    """
    edge task:
    label1, label1
    label2, label2
    label3, label3
    """
    if dataname is None:
        raise KeyError("dataname is None!")

    index_path = './dataset/{}/index/'.format(dataname.lower())
    mkdir(index_path)

    node_labels = data.ndata['y'][targetnode]

    for n_label in range(node_classes):
        """
        node-task: [0, num_node_classes)
        edge-task: [num_node_classes, 2*num_node_classes)
        """
        task_id = node_classes + n_label

        subset = torch.argwhere(node_labels == n_label)  # (num, 1)


        subset = subset[torch.randperm(subset.shape[0])]
        # TODO: ensure each label contain more than 400 nodes

        if subset.shape[0] < 400:
            warnings.warn("label {} only has {} nodes but it should be larger than 400!".format(task_id, subset.shape[0]),
                          RuntimeWarning)
        else:
            subset = subset[0:400]


        sub_edges=[]
        for node in subset:
            # 为给定节点创建一个k跳入度子图
            sub_graph, _ = dgl.khop_in_subgraph(graph=data, nodes={targetnode: node}, k=1)

            # 筛选出在子图中有边的有效边类型
            valid_etypes = [etype for etype in sub_graph.canonical_etypes if len(sub_graph.edata[dgl.EID][etype]) > 0]

            # 从有效边类型中选择一个随机边类型
            if valid_etypes:
                etype = random.choice(valid_etypes)
                edge_id = random.choice(sub_graph.edata[dgl.EID][etype])
                edge = {etype: edge_id}
                sub_edges.append(edge)
            else:
                print(f"节点 {node} 未找到有效的边类型。")

        print("label {} total sub_edges {}".format(n_label, len(sub_edges)))

        # TODO: you can also sample even more edges (larger than 400)
        edge_index_100_shot = sub_edges[0:400]
        # print(edge_index_400_shot.shape)

        dname = 'task{}'.format(task_id)

        pk.dump(edge_index_100_shot,
                    open(index_path + dname, 'bw'))

def induced_graphs_nodes(data, dataname: str = None, num_classes=3, smallest_size=100, largest_size=300,targetnode=None,feats_type=None,hop_num=None):
    """
    node-level: [0,num_classes)
    edge-level: [num_classes,num_classes*2)
    graph-level: [num_classes*2,num_classes*3)
    """
    if dataname is None:
        raise KeyError("dataname is None!")

    induced_graphs_path = './dataset/{}/induced_graphs/'.format(dataname.lower())
    mkdir(induced_graphs_path)


    #ori_x = data.ndata['x'][targetnode]

    fnames = []
    for i in range(0, num_classes):  # TODO: remember to reset to num_classies!
        fname = './dataset/{}/index/task{}'.format(dataname.lower(),i)
        fnames.append(fname)

    for fname in fnames:
        i = int(fname.split('/')[-1][4:])
        a = pk.load(open(fname, 'br'))
        value = a

        value = value[torch.randperm(value.shape[0])]

        iteration = 0
        induced_graph_dic_list = defaultdict(list)
        induced_graph_list = []

        if(hop_num!=0):
            #HGPROMPT的子图采样 固定1hop
            for node in torch.flatten(value):
                iteration = iteration + 1
                subgraph, inverse_indices = dgl.khop_in_subgraph(data, {targetnode: node}, k=hop_num)
                num_nodes = subgraph.num_nodes()

                if (dataname == "IMDB"):
                    label = data.ndata['oldy'][targetnode][node]
                    if (torch.any(label)):
                        induced_graph_list.append((subgraph,inverse_indices,label))
                else:
                    induced_graph_list.append((subgraph,inverse_indices))
                print(
                    'graph size {} at {:.2f}%...'.format(subgraph.num_nodes(), iteration * 100.0 / value.shape[0]))

            induced_graph_dic_list['pos'] = induced_graph_list

            if len(induced_graph_dic_list['pos']) < 400:
                # raise ValueError("candidate graphs should be at least 400")
                warnings.warn("===task{} has not enough graphs "
                              "(should be 400 but got {})".format(i, len(induced_graph_dic_list['pos'])),
                              RuntimeWarning)

            pk.dump(induced_graph_dic_list,
                    open('{}task{}.hop{}.ft{}'.format(induced_graphs_path, i,hop_num,feats_type), 'bw'))

        else:
        #本论文的子图采集算法
            for node in torch.flatten(value):

                iteration = iteration + 1

                subgraph, inverse_indices = dgl.khop_in_subgraph(data,{targetnode:node},k=1)
                current_hop = 1
                num_nodes=subgraph.num_nodes()
                while num_nodes < smallest_size and current_hop < 5:
                    # print("subgraph smaller than {} explore higher hop...".format(smallest_size))
                    current_hop = current_hop + 1
                    subgraph, inverse_indices = dgl.khop_in_subgraph(data,{targetnode:node},k=current_hop)
                    num_nodes = subgraph.num_nodes()

                if num_nodes > largest_size:
                    target_total_size = largest_size  # 指定总大小
                    specified_nodes = inverse_indices  # 指定要保留的节点
                    proportional_sizes = calculate_proportional_size(subgraph, target_total_size)
                    sampled_nodes = sample_nodes(subgraph, proportional_sizes, specified_nodes)
                    subgraph = subgraph_from_nodes(subgraph, sampled_nodes)

                if(subgraph.num_nodes()>=smallest_size and subgraph.num_nodes()<=largest_size+10):
                    if(dataname=="IMDB"):
                        label=data.ndata['oldy'][targetnode][node]
                        if(torch.any(label)):
                            induced_graph_list.append((subgraph,label))
                    else:
                        induced_graph_list.append(subgraph)
                    print('graph size {} at {:.2f}%...'.format(subgraph.num_nodes(), iteration * 100.0 / value.shape[0]))

            induced_graph_dic_list['pos'] = induced_graph_list

            if len(induced_graph_dic_list['pos']) < 400:
                # raise ValueError("candidate graphs should be at least 400")
                warnings.warn("===task{} has not enough graphs "
                              "(should be 400 but got {})".format(i, len(induced_graph_dic_list['pos'])),
                              RuntimeWarning)

            pk.dump(induced_graph_dic_list,
                    open('{}task{}.hop{}.ft{}'.format(induced_graphs_path, i,hop_num,feats_type), 'bw'))

        print('node-induced graphs saved!')


def induced_graphs_edges(data, dataname: str = None, num_classes=3, smallest_size=100, largest_size=300,targetnode=None,feats_type=None,hop_num=1):
    """
        node-level: [0,num_classes)
        edge-level: [num_classes,num_classes*2)
        graph-level: [num_classes*2,num_classes*3)
    """
    if dataname is None:
        raise KeyError("dataname is None!")

    induced_graphs_path = './dataset/{}/induced_graphs/'.format(dataname.lower())
    mkdir(induced_graphs_path)



    fnames = []
    for task_id in range(num_classes, 2 * num_classes):
        fname = './dataset/{}/index/task{}'.format(dataname.lower(), task_id)
        fnames.append(fname)

    # 1-hop edge induced graphs
    for fname in fnames:
        induced_graph_dic_list = defaultdict(list)

        sp = fname.split('.')
        prefix_task_id= sp[-1]
        task_id = int(prefix_task_id.split('/')[-1][4:])

        n_label = task_id - num_classes

        value = pk.load(open(fname, 'br'))

        induced_graph_list = []
        iteration = 0
        for c in range(len(value)):
            iteration = iteration + 1

            edge=value[c]
            etype,eid = list(edge.keys())[0], list(edge.values())[0]
            edges=data.edges(etype=etype)
            eid=eid.item()
            src_n, tar_n = edges[0][eid].item(), edges[1][eid].item()
            src_type,tar_type = etype[0],etype[-1]


            # subgraph,inverse_indices=dgl.khop_in_subgraph(data,{src_type:src_n,tar_type:tar_n},k=1)
            # num_nodes=subgraph.num_nodes()
            # temp_hop = 1
            # while num_nodes < smallest_size and temp_hop < 3:
            #     # print("subset smaller than {} explore higher hop...".format(smallest_size))
            #     temp_hop = temp_hop + 1
            #     subgraph,inverse_indices=dgl.khop_in_subgraph(data,{src_type:src_n,tar_type:tar_n},k=temp_hop)
            #     num_nodes = subgraph.num_nodes()
            #
            # if num_nodes > largest_size:
            #     target_total_size = largest_size  # 指定总大小
            #     specified_nodes = inverse_indices  # 指定要保留的节点
            #     proportional_sizes = calculate_proportional_size(subgraph, target_total_size)
            #     sampled_nodes = sample_nodes(subgraph, proportional_sizes, specified_nodes)
            #     subgraph = subgraph_from_nodes(subgraph, sampled_nodes)
            #
            #
            # if (subgraph.num_nodes() > smallest_size and subgraph.num_nodes() < largest_size+10):
            #     if(dataname=="IMDB"):
            #         if(targetnode==tar_type):
            #             node=tar_n
            #         else:
            #             node=src_n
            #         label = data.ndata['oldy'][targetnode][node]
            #         if (torch.any(label)):
            #             induced_graph_list.append((subgraph, label))
            #     else:
            #         induced_graph_list.append(subgraph)
            subgraph, inverse_indices = dgl.khop_in_subgraph(data, {src_type: src_n, tar_type: tar_n}, k=hop_num)

            if (dataname == "IMDB"):
                if (targetnode == tar_type):
                    node = tar_n
                else:
                    node = src_n
                label = data.ndata['oldy'][targetnode][node]
                if (torch.any(label)):
                    induced_graph_list.append((subgraph, label))
            else:
                induced_graph_list.append(subgraph)
            print('graph size {} at {:.2f}%...'.format(subgraph.num_nodes(), iteration * 100.0 / len(value)))

        induced_graph_dic_list['pos'] = induced_graph_list

        pk.dump(induced_graph_dic_list,
                open('{}task{}.hop{}.ft{}'.format(induced_graphs_path, task_id,hop_num,feats_type), 'bw'))


def induced_graphs_graphs(data, dataname: str = None, num_classes=3, smallest_size=100,largest_size=300,targetnode=None,feats_type=None,hop_num=1):
    """
        node-level: [0,num_classes)
        edge-level: [num_classes,num_classes*2)
        graph-level: [num_classes*2,num_classes*3)

    可否这样做graph induced graph？
    metis生成多个graph
    然后对这些graph做扰动变成更多的graphs
    """
    if dataname is None:
        raise KeyError("dataname is None!")

    induced_graphs_path = './dataset/{}/induced_graphs/'.format(dataname.lower())
    mkdir(induced_graphs_path)

    node_labels = data.ndata['y'][targetnode]


    for n_label in range(num_classes):
        task_id = 2 * num_classes + n_label

        nodes = torch.squeeze(torch.argwhere(node_labels == n_label))
        nodes = nodes[torch.randperm(nodes.shape[0])]
        # print("there are {} nodes for label {} task_id {}".format(nodes.shape[0],n_label,task_id))


        # # I previouly use the following to construct graph but most of the baselines ouput 1.0 acc.
        # same_label_edge_index, _ = subgraph(nodes, edge_index, num_nodes=num_nodes,
        #                                     relabel_nodes=False)  # attention! relabel_nodes=False!!!!!!

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



        dname = 'task{}'.format(task_id)

        induced_graph_dic_list = defaultdict(list)
        iteration = 0

        for seeds in seeds_list:
            # iteration = iteration + 1
            # nodeset={targetnode:seeds}
            # subgraph,inverse_indices=dgl.khop_in_subgraph(data,nodeset,k=1)
            #
            # # regularize its size
            #
            # temp_hop = 1
            # while subgraph.num_nodes() < smallest_size and temp_hop < 5:
            #     temp_hop = temp_hop + 1
            #     subgraph, inverse_indices = dgl.khop_in_subgraph(data, nodeset, k=temp_hop)
            #
            #
            # if subgraph.num_nodes() > largest_size:
            #     target_total_size = largest_size  # 指定总大小
            #     specified_nodes = inverse_indices  # 指定要保留的节点
            #     proportional_sizes = calculate_proportional_size(subgraph, target_total_size)
            #     sampled_nodes = sample_nodes(subgraph, proportional_sizes, specified_nodes)
            #     subgraph = subgraph_from_nodes(subgraph, sampled_nodes)
            #
            # if (subgraph.num_nodes() > smallest_size and subgraph.num_nodes() < largest_size+10):
            #     induced_graph_dic_list['pos'].append(subgraph)
            #     print(
            #     'graph size {} at {:.2f}%...'.format(subgraph.num_nodes(), iteration * 100.0 / len(seeds_list)))
            iteration = iteration + 1
            nodeset = {targetnode: seeds}
            subgraph, inverse_indices = dgl.khop_in_subgraph(data, nodeset, k=hop_num)


            induced_graph_dic_list['pos'].append(subgraph)
            print(
                'graph size {} at {:.2f}%...'.format(subgraph.num_nodes(), iteration * 100.0 / len(seeds_list)))

        pk.dump(induced_graph_dic_list,
                open('{}{}.hop{}.ft{}'.format(induced_graphs_path, dname,hop_num,feats_type), 'bw'))

        print("{} saved! len {}".format(dname, len(induced_graph_dic_list['pos'])))


def induced_graph_2_K_shot(t1_dic, t2_dic, dataname: str = None,
                           K=None, seed=None):
    if dataname is None:
        raise KeyError("dataname is None!")
    if K:
        t1_pos = t1_dic['pos'][0:K]
        t2_pos = t2_dic['pos'][0:K]  # treat as neg
    else:
        t1_pos = t1_dic['pos']
        t2_pos = t2_dic['pos']  # treat as neg

    task_data = []
    for g in t1_pos:
        g.y = torch.tensor([1]).long()
        task_data.append(g)

    for g in t2_pos:
        g.y = torch.tensor([0]).long()
        task_data.append(g)

    if seed:
        random.seed(seed)
    random.shuffle(task_data)

    batch = Batch.from_data_list(task_data)

    return batch


def load_tasks(meta_stage: str, task_pairs: list, dataname: str = None, K_shot=None, seed=0):
    if dataname is None:
        raise KeyError("dataname is None!")

    """
    :param meta_stage: 'train', 'test'
    :param task_id_list:
    :param K_shot:  default: None.
                    if K_shot is None, load the full data to train/test meta.
                    Else: K-shot learning with 2*K graphs (pos:neg=1:1)
    :param seed:
    :return: iterable object of (task_id, support, query)


    # 从序列中取2个元素进行排列
        for e in it.permutations('ABCD', 2):
            print(''.join(e), end=', ') # AB, AC, AD, BA, BC, BD, CA, CB, CD, DA, DB, DC,

    # 从序列中取2个元素进行组合、元素不允许重复
        for e in it.combinations('ABCD', 2):
            print(''.join(e), end=', ') # AB, AC, AD, BC, BD, CD,

    """

    max_iteration = 100

    i = 0
    while i < len(task_pairs) and i < max_iteration:
        task_1, task_2 = task_pairs[i]

        task_1_support = './dataset/{}/induced_graphs/task{}.meta.{}.support'.format(dataname, task_1, meta_stage)
        task_1_query = './dataset/{}/induced_graphs/task{}.meta.{}.query'.format(dataname, task_1, meta_stage)
        task_2_support = './dataset/{}/induced_graphs/task{}.meta.{}.support'.format(dataname, task_2, meta_stage)
        task_2_query = './dataset/{}/induced_graphs/task{}.meta.{}.query'.format(dataname, task_2, meta_stage)

        with open(task_1_support, 'br') as t1s,\
                open(task_1_query, 'br') as t1q,\
                open(task_2_support, 'br') as t2s,\
                open(task_2_query, 'br') as t2q:
            t1s_dic, t2s_dic = pk.load(t1s), pk.load(t2s)
            support = induced_graph_2_K_shot(t1s_dic, t2s_dic, dataname, K=K_shot, seed=seed)

            t1q_dic, t2q_dic = pk.load(t1q), pk.load(t2q)
            query = induced_graph_2_K_shot(t1q_dic, t2q_dic, dataname, K=K_shot, seed=seed)

        i = i + 1
        yield task_1, task_2, support, query, len(task_pairs)

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
            if num_specified >= size:
                sampled_nodes[ntype] = torch.tensor(specified_nodes_ntype)
            else:
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



if __name__ == '__main__':

    #heco,dmgi分别为预训练好的embedding
    #6就是HeCo的embeding 7就是DMGI, -1指的是freebase的无向图情况 8 就是hdmi 9 HGMAE
    hop_num_dict = {'ACM': 1,
               'DBLP': 2,
               'IMDB': 2,
               'Freebase': 1,
               'oldfreebase': 2}
    feats_type=0

    dataname = 'IMDB' #'IMDB'  'ACM'  'DBLP'  'Freebase'
    hop_num=hop_num_dict[dataname]

    if(dataname=='Freebase' and feats_type==-1):
        dataset = HGBDataset(root='./dataset', name=dataname,transform=ToUndirected(merge=False))
        data = dataset[0]
    elif(dataname=='oldfreebase'):
        data=load_freebase('./dataset/oldfreebase')
    else:
        dataset = HGBDataset(root='./dataset', name=dataname)
        data=dataset[0]


    for node_type, node_store in data.node_items():
        for attr, value in node_store.items():
            #没有节点属性的节点类型赋值对角矩阵
            if attr=='num_nodes':
                #data[node_type]['x']=torch.eye(value)
                data[node_type]['x'] = create_matrix(value,0.01)
                del data[node_type][attr]


    features=data.x_dict
    features_list=[]
    for value in features.values():
        features_list.append(value)

    if feats_type == 0 or feats_type == 6 or feats_type == 7 or feats_type == -1 or feats_type == 8 or feats_type == 9:
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
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to_dense()
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to_dense()


    if(feats_type==6):
        embed=torch.load('./dataset/'+dataname.lower()+'/HeCo.pth')
        features_list[0]=embed
    elif(feats_type==7):
        embed = torch.load('./dataset/' + dataname.lower() + '/DMGI.pth')
        features_list[0] = embed
    # elif (feats_type == 8):
    #     embed = torch.load('./dataset/' + dataname.lower() + '/HDMI.pth')
    #     features_list[0] = embed.to('cpu')
    # elif (feats_type == 9):
    #     embed = torch.load('./dataset/' + dataname.lower() + '/HGMAE.pth')
    #     features_list[0] = embed

    value_dict={}
    i = 0
    for ntype in data.node_types:
        value_dict[ntype]=features_list[i]
        i=i+1


    data.set_value_dict('x',value_dict)

    data=to_dgl(data)

    if dataname=='IMDB':
        targetnode='movie'
        oldy=data.ndata['y'][targetnode]
        #newy=data.ndata['y'][targetnode].argmax(axis=1)
        newy = []
        for arr in data.ndata['y'][targetnode]:
            # 获取所有值为 1 的下标
            one_indices = np.where(arr == 1)[0]
            # 检查是否有值为 1 的元素
            if one_indices.size > 0:
                # 随机选择一个值为 1 的下标
                random_index = np.random.choice(one_indices)
                newy.append(random_index)
            else:
                # 如果没有值为 1 的元素，可以选择填充一个特殊值（比如 -1）
                newy.append(-1)
        newy=torch.tensor(newy)
        data.ndata['y']={targetnode:newy}
        data.ndata['oldy'] = {targetnode:oldy}
        node_classes = torch.max(data.ndata['y'][targetnode])+1

    elif dataname=='DBLP':
        targetnode = 'author'
        node_classes = torch.max(data.ndata['y'][targetnode])+1
    elif dataname=='ACM':
        targetnode = 'paper'
        node_classes = torch.max(data.ndata['y'][targetnode])+1
    elif dataname=='Freebase':
        targetnode= 'book'
        node_classes = torch.max(data.ndata['y'][targetnode])+1
    elif dataname == 'oldfreebase':
        targetnode = 'movie'
        node_classes = torch.max(data.ndata['y'][targetnode]) + 1

    # # step1 split node and edge
    #
    # nodes_split(data, dataname=dataname, node_classes=node_classes,targetnode=targetnode)
    # edge_split(data, dataname=dataname, node_classes=node_classes,targetnode=targetnode)

    # # step2: induced graphs
    induced_graphs_nodes(data, dataname=dataname, num_classes=node_classes, smallest_size=50,
                         largest_size=200,targetnode=targetnode,feats_type=feats_type,hop_num=hop_num)
    induced_graphs_edges(data, dataname=dataname, num_classes=node_classes, smallest_size=50,
                         largest_size=500,targetnode=targetnode,feats_type=feats_type,hop_num=hop_num)
    induced_graphs_graphs(data, dataname=dataname, num_classes=node_classes, smallest_size=50,
                          largest_size=300,targetnode=targetnode,feats_type=feats_type,hop_num=hop_num)

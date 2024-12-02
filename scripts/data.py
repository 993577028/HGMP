import numpy as np
import torch
import pickle as pk
from random import shuffle
import os


from collections import defaultdict
import dgl


def multi_class_NIG(dataname, num_class,shots=100,classification_type=0,feats_type=None):
    """
    NIG: node induced graphs
    :param dataname: e.g. CiteSeer
    :param num_class: e.g. 6 (for CiteSeer)
    :return: a batch of training graphs and testing graphs
    """
    hop_num={'ACM':1,
             'DBLP':2,
             'IMDB':2,
             'Freebase':1,
             'oldfreebase':2}
    hop_num=hop_num[dataname]
    if classification_type == 'NIG':
        flag=0
    elif classification_type == 'EIG':
        flag=1
    elif classification_type == 'GIG':
        flag=2
    else:
        raise ValueError("model hasn't supported {} task".format(classification_type))
    statistic = defaultdict(list)

    # load training NIG (node induced graphs)
    train_list = []
    valid_list = []
    test_list = []
    for task_id in range(flag*num_class,num_class*(flag+1)):

        data_list = list()

        data_path = './dataset/{}/induced_graphs/task{}.hop{}.ft{}'.format(dataname.lower(),task_id,hop_num,feats_type)
        #data_path = './dataset/{}/induced_graphs/task{}.{}'.format(dataname.lower(), task_id, feats_type)

        with open(data_path, 'br') as f:
            data_list = pk.load(f)['pos']

        shuffle(data_list)
        data_list1 = data_list[0:shots]
        data_list2 = data_list[shots:shots*2]
        data_list3 = data_list[shots * 2:-1]
        statistic['train'].append((task_id, len(data_list1)))
        statistic['valid'].append((task_id, len(data_list2)))
        statistic['test'].append((task_id, len(data_list3)))

        label=task_id-flag*num_class
        if(classification_type=='NIG'):
            if(dataname=='IMDB'):
                for g in data_list1:
                    train_list.append(g)
                for g in data_list2:
                    valid_list.append(g)
                for g in data_list3:
                    test_list.append(g)
            else:
                for g in data_list1:
                    train_list.append((g[0],g[1],torch.tensor(label)))
                for g in data_list2:
                    valid_list.append((g[0],g[1],torch.tensor(label)))
                for g in data_list3:
                    test_list.append((g[0],g[1],torch.tensor(label)))
        elif(classification_type=='EIG'):
            if (dataname == 'IMDB'):
                for g in data_list1:
                    train_list.append(g)
                for g in data_list2:
                    valid_list.append(g)
                for g in data_list3:
                    test_list.append(g)
            else:
                for g in data_list1:
                    train_list.append((g, torch.tensor(label)))
                for g in data_list2:
                    valid_list.append((g, torch.tensor(label)))
                for g in data_list3:
                    test_list.append((g, torch.tensor(label)))
        elif (classification_type == 'GIG'):
            for g in data_list1:
                train_list.append((g, torch.tensor(label)))
            for g in data_list2:
                valid_list.append((g, torch.tensor(label)))
            for g in data_list3:
                test_list.append((g, torch.tensor(label)))
    shuffle(train_list)
    # train_data=dgl.batch(train_list)
    #train_data = Batch.from_data_list(train_list)

    shuffle(valid_list)
    shuffle(test_list)
    # test_data = dgl.batch(test_list)
    #test_data = Batch.from_data_list(test_list)

    # for key, value in statistic.items():
    #     print("{}ing set (class_id, graph_num): {}".format(key, value))

    return train_list,valid_list,test_list

def load_nodes(dataname, num_classes,shots=100):

    statistic = defaultdict(list)
    train_index=[]
    val_index=[]
    test_index=[]
    fnames = []
    for i in range(0, num_classes):  # TODO: remember to reset to num_classies!
        fname = '../dataset/{}/index/task{}'.format(dataname.lower(), i)
        fnames.append(fname)

    for fname in fnames:
        i = int(fname.split('/')[-1][4:])
        value = pk.load(open(fname, 'br'))
        value = value[torch.randperm(value.shape[0])]

        data_list1 = value[0:shots]
        data_list2 = value[shots:shots * 2]
        data_list3 = value[shots * 2:-1]
        statistic['train'].append((i, len(data_list1)))
        statistic['valid'].append((i, len(data_list2)))
        statistic['test'].append((i, len(data_list3)))

        for index in data_list1:
            train_index.append(index.item())
        for index in data_list2:
            val_index.append(index.item())
        for index in data_list3:
            test_index.append(index.item())

    shuffle(train_index)
    shuffle(val_index)
    shuffle(test_index)

    train_index=np.array(train_index)
    val_index = np.array(val_index)
    test_index = np.array(test_index)

    for key, value in statistic.items():
        print("{}ing set (class_id, graph_num): {}".format(key, value))

    pk.dump(train_index, open('../dataset/{}/'.format(dataname.lower()) + "train_index_{}shots".format(shots), 'bw'))
    pk.dump(val_index, open('../dataset/{}/'.format(dataname.lower()) + "val_index_{}shots".format(shots), 'bw'))
    pk.dump(test_index, open('../dataset/{}/'.format(dataname.lower()) + "test_index_{}shots".format(shots), 'bw'))

    return train_index, val_index, test_index

if __name__ == '__main__':

    #multi_class_NIG('IMDB',5,100)
    load_nodes('IMDB', 5, shots=10)

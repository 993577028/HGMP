import torch
import numpy as np
import torchmetrics
import warnings
import dgl
from HGP.utils import graph_pool


class Evaluator:
    def __init__(self, eval_metric='hits@50'):

        self.eval_metric = eval_metric
        if 'hits@' in self.eval_metric:
            self.K = int(self.eval_metric.split('@')[1])

    def _parse_and_check_input(self, input_dict):
        if 'hits@' in self.eval_metric:
            if not 'y_pred_pos' in input_dict:
                raise RuntimeError('Missing key of y_pred_pos')
            if not 'y_pred_neg' in input_dict:
                raise RuntimeError('Missing key of y_pred_neg')

            y_pred_pos, y_pred_neg = input_dict['y_pred_pos'], input_dict['y_pred_neg']

            if not (isinstance(y_pred_pos, np.ndarray) or (torch is not None and isinstance(y_pred_pos, torch.Tensor))):
                raise ValueError('y_pred_pos needs to be either numpy ndarray or torch tensor')

            if not (isinstance(y_pred_neg, np.ndarray) or (torch is not None and isinstance(y_pred_neg, torch.Tensor))):
                raise ValueError('y_pred_neg needs to be either numpy ndarray or torch tensor')

            if torch is not None and (isinstance(y_pred_pos, torch.Tensor) or isinstance(y_pred_neg, torch.Tensor)):
                if isinstance(y_pred_pos, np.ndarray):
                    y_pred_pos = torch.from_numpy(y_pred_pos)

                if isinstance(y_pred_neg, np.ndarray):
                    y_pred_neg = torch.from_numpy(y_pred_neg)

                y_pred_pos = y_pred_pos.to(y_pred_neg.device)

                type_info = 'torch'

            else:
                type_info = 'numpy'

            if not y_pred_pos.ndim == 1:
                raise RuntimeError('y_pred_pos must to 1-dim arrray, {}-dim array given'.format(y_pred_pos.ndim))

            return y_pred_pos, y_pred_neg, type_info

        elif 'mrr' == self.eval_metric:

            if not 'y_pred_pos' in input_dict:
                raise RuntimeError('Missing key of y_pred_pos')
            if not 'y_pred_neg' in input_dict:
                raise RuntimeError('Missing key of y_pred_neg')

            y_pred_pos, y_pred_neg = input_dict['y_pred_pos'], input_dict['y_pred_neg']

            if not (isinstance(y_pred_pos, np.ndarray) or (torch is not None and isinstance(y_pred_pos, torch.Tensor))):
                raise ValueError('y_pred_pos needs to be either numpy ndarray or torch tensor')

            if not (isinstance(y_pred_neg, np.ndarray) or (torch is not None and isinstance(y_pred_neg, torch.Tensor))):
                raise ValueError('y_pred_neg needs to be either numpy ndarray or torch tensor')

            if torch is not None and (isinstance(y_pred_pos, torch.Tensor) or isinstance(y_pred_neg, torch.Tensor)):
                if isinstance(y_pred_pos, np.ndarray):
                    y_pred_pos = torch.from_numpy(y_pred_pos)

                if isinstance(y_pred_neg, np.ndarray):
                    y_pred_neg = torch.from_numpy(y_pred_neg)

                y_pred_pos = y_pred_pos.to(y_pred_neg.device)

                type_info = 'torch'
            else:
                type_info = 'numpy'

            if not y_pred_pos.ndim == 1:
                raise RuntimeError('y_pred_pos must to 1-dim arrray, {}-dim array given'.format(y_pred_pos.ndim))

            if not y_pred_neg.ndim == 2:
                raise RuntimeError('y_pred_neg must to 2-dim arrray, {}-dim array given'.format(y_pred_neg.ndim))

            return y_pred_pos, y_pred_neg, type_info

        else:
            raise ValueError('Undefined eval metric %s' % (self.eval_metric))

    def eval(self, input_dict):

        if 'hits@' in self.eval_metric:
            y_pred_pos, y_pred_neg, type_info = self._parse_and_check_input(input_dict)
            return self._eval_hits(y_pred_pos, y_pred_neg, type_info)
        elif self.eval_metric == 'mrr':
            y_pred_pos, y_pred_neg, type_info = self._parse_and_check_input(input_dict)
            return self._eval_mrr(y_pred_pos, y_pred_neg, type_info)

        else:
            raise ValueError('Undefined eval metric %s' % (self.eval_metric))

    def _eval_hits(self, y_pred_pos, y_pred_neg, type_info):

        if type_info == 'torch':
            res = torch.topk(y_pred_neg, self.K)
            kth_score_in_negative_edges = res[0][:, -1]
            hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)

        else:
            kth_score_in_negative_edges = np.sort(y_pred_neg)[-self.K]
            hitsK = float(np.sum(y_pred_pos > kth_score_in_negative_edges)) / len(y_pred_pos)

        return {'hits@{}'.format(self.K): hitsK}

    def _eval_mrr(self, y_pred_pos, y_pred_neg, type_info):
        if type_info == 'torch':
            y_pred = torch.cat([y_pred_pos.view(-1, 1), y_pred_neg], dim=1)
            argsort = torch.argsort(y_pred, dim=1, descending=True)
            ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
            ranking_list = ranking_list[:, 1] + 1
            mrr_list = 1. / ranking_list.to(torch.float)
            return mrr_list.mean()
        else:
            y_pred = np.concatenate([y_pred_pos.reshape(-1, 1), y_pred_neg], axis=1)
            argsort = np.argsort(-y_pred, axis=1)
            ranking_list = (argsort == 0).nonzero()
            ranking_list = ranking_list[1] + 1
            mrr_list = 1. / ranking_list.astype(np.float32)
            return mrr_list.mean()


def mrr_hit(normal_label: np.ndarray, pos_out: np.ndarray, metric: list = None):
    if isinstance(normal_label, np.ndarray) and isinstance(pos_out, np.ndarray):
        pass
    else:
        warnings.warn('it would be better if normal_label and out are all set as np.ndarray')

    results = {}
    if not metric:
        metric = ['mrr', 'hits']

    if 'hits' in metric:
        hits_evaluator = Evaluator(eval_metric='hits@50')
        flag = normal_label
        pos_test_pred = torch.from_numpy(pos_out[flag == 1])
        neg_test_pred = torch.from_numpy(pos_out[flag == 0])

        for N in [100]:
            neg_test_pred_N = neg_test_pred.view(-1, 100)
            for K in [1, 5, 10]:
                hits_evaluator.K = K
                test_hits = hits_evaluator.eval({
                    'y_pred_pos': pos_test_pred,
                    'y_pred_neg': neg_test_pred_N,
                })[f'hits@{K}']

                results[f'Hits@{K}@{N}'] = test_hits

    if 'mrr' in metric:
        mrr_evaluator = Evaluator(eval_metric='mrr')
        flag = normal_label
        pos_test_pred = torch.from_numpy(pos_out[flag == 1])
        neg_test_pred = torch.from_numpy(pos_out[flag == 0])

        neg_test_pred = neg_test_pred.view(-1, 100)

        mrr = mrr_evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })

        if isinstance(mrr, torch.Tensor):
            mrr = mrr.item()
        results['mrr'] = mrr
    return results


def acc_f1_over_batches(test_loader, PG, hgnn, answering, num_class, task_type, device,targetnode,dataname,classification_type):
    PG = PG.to("cpu")
    if answering is not None:
        answering = answering.to("cpu")
    hgnn = hgnn.to("cpu")
    if task_type == "multi_class_classification":
        micro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="micro")
        macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro")
    elif task_type == "binary_classification":
        micro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="micro")
        macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro")
    else:
        raise NotImplementedError

    for batch_id, test_batch in enumerate(test_loader):
        if (classification_type == 'NIG'):
            batched_graph, _, batched_label = test_batch
        else:
            batched_graph, batched_label = test_batch


        prompted_graph = PG(batched_graph)

        x_dict = prompted_graph.ndata['x']
        edge_index_dict = {}
        for etype in prompted_graph.canonical_etypes:
            edge_index = prompted_graph.edges(etype=etype)
            src = edge_index[0].unsqueeze(0)
            tar = edge_index[1].unsqueeze(0)
            edge_index = torch.cat((src, tar), dim=0)
            edge_index_dict[etype] = edge_index

        if (hgnn.hgnn_type == 'HGT'):
            node_emb = hgnn(targetnode, x_dict, edge_index_dict)
        elif (hgnn.hgnn_type == 'SHGN'):
            node_emb = hgnn(targetnode, prompted_graph, x_dict)
        elif (hgnn.hgnn_type == 'GCN'):
            node_emb = hgnn(prompted_graph, x_dict)
        elif (hgnn.hgnn_type == 'GAT'):
            node_emb = hgnn(prompted_graph, x_dict, False)

        graph_emb = graph_pool('mean', node_emb, prompted_graph)

        pre = answering(graph_emb)

        pre = pre.detach()
        y = batched_label

        pre_cla = torch.argmax(pre, dim=1)
        if(dataname=="IMDB" and classification_type!='GIG'):
            pre_cla=torch.nn.functional.one_hot(pre_cla, num_classes=num_class)

        mi_f1 = micro_f1(pre_cla, y)
        ma_f1 = macro_f1(pre_cla, y)
        #print("Batch {} Acc: {:.4f} | Macro-F1: {:.4f}".format(batch_id, acc.item(), ma_f1.item()))

    mi_f1 = micro_f1.compute()
    ma_f1 = macro_f1.compute()
    #print("Final True Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(mi_f1.item(), ma_f1.item()))
    micro_f1.reset()
    macro_f1.reset()
    PG = PG.to(device)
    if answering is not None:
        answering = answering.to(device)
    hgnn = hgnn.to(device)
    return mi_f1.item(), ma_f1.item()

def valid_over_batches(valid_loader, PG, hgnn, answering, num_class, task_type, device,targetnode,dataname,lossfn,classification_type):
    PG = PG.to("cpu")
    if answering is not None:
        answering = answering.to("cpu")
    hgnn = hgnn.to("cpu")
    if task_type == "multi_class_classification":
        micro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="micro")
        macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro")
    elif task_type == "binary_classification":
        micro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="micro")
        macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro")
    else:
        raise NotImplementedError

    total_loss=0
    for batch_id, test_batch in enumerate(valid_loader):
        if(classification_type=='NIG'):
            batched_graph,_,batched_label=test_batch
        else:
            batched_graph,batched_label = test_batch


        prompted_graph = PG(batched_graph)


        x_dict = prompted_graph.ndata['x']
        edge_index_dict = {}
        for etype in prompted_graph.canonical_etypes:
            edge_index = prompted_graph.edges(etype=etype)
            src = edge_index[0].unsqueeze(0)
            tar = edge_index[1].unsqueeze(0)
            edge_index = torch.cat((src, tar), dim=0)
            edge_index_dict[etype] = edge_index

        if (hgnn.hgnn_type == 'HGT'):
            node_emb = hgnn(targetnode, x_dict, edge_index_dict)
        elif (hgnn.hgnn_type == 'SHGN'):
            node_emb = hgnn(targetnode, prompted_graph, x_dict)
        elif (hgnn.hgnn_type == 'GCN'):
            node_emb = hgnn(prompted_graph, x_dict)
        elif (hgnn.hgnn_type == 'GAT'):
            node_emb = hgnn(prompted_graph, x_dict, False)


        graph_emb = graph_pool('mean', node_emb, prompted_graph)


        pre = answering(graph_emb)
        pre = pre.detach()
        y = batched_label

        valid_loss=lossfn(pre,y)
        total_loss+=valid_loss.item()

        pre_cla = torch.argmax(pre, dim=1)
        if(dataname=="IMDB" and classification_type!='GIG'):
            pre_cla=torch.nn.functional.one_hot(pre_cla, num_classes=num_class)

        mi_f1 = micro_f1(pre_cla, y)
        ma_f1 = macro_f1(pre_cla, y)
        #print("Batch {} Acc: {:.4f} | Macro-F1: {:.4f}".format(batch_id, acc.item(), ma_f1.item()))

    mi_f1 = micro_f1.compute()
    ma_f1 = macro_f1.compute()
    mean_loss=total_loss/len(valid_loader)
    #print("valid loss: {:.8f} | Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(mean_loss,mi_f1.item(), ma_f1.item()))
    micro_f1.reset()
    macro_f1.reset()
    PG = PG.to(device)
    if answering is not None:
        answering = answering.to(device)
    hgnn = hgnn.to(device)
    return mean_loss

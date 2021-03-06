# !/usr/bin/env python
# coding: utf-8


import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import time
import numpy as np
import sys

sys.setrecursionlimit(50000)
import pickle
import random

torch.nn.Module.dump_patches = True
import copy
import pandas as pd
from AttentiveFP.featurizing import graph_dict as graph
from AttentiveFP.AttentiveLayers import Fingerprint, graph_dataset, null_collate, Graph, Logger, time_to_str
from rdkit import Chem
import json
SEED = 168
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True





class Attentivefp(object):
    def __init__(self, filename, task_conf,**kwargs):
        self.batch_size = 50
        self.epochs = 200
        self.p_dropout = 0.2
        self.fingerprint_dim = 128
        self.weight_decay = 5  # also known as l2_regularization_lambda
        self.learning_rate = 3.5
        self.K = 2
        self.T = 2
        self.label_class = None
        self.model_path = 'best'
        self.weighted = 'mean'
        self.gpu = False

        try:
            self.type = task_conf['type']
            self.label_class = task_conf['label_class']
            self.task_name = filename
        except:
            raise KeyError("read {} task conf error".format(filename))

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        if self.gpu :
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else :
            self.device = 'cpu'

    @staticmethod
    def pre_data(smiles_list):
        """
        Checking molecular legitimacy

        :param smiles_list: smiles list of molecules
        :return:
        """
        print("number of all smiles: ", len(smiles_list))
        atom_num_dist = []
        remained_smiles = []
        canonical_smiles_list = []
        del_smiles_list = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                atom_num_dist.append(len(mol.GetAtoms()))
                Chem.SanitizeMol(mol)
                Chem.DetectBondStereochemistry(mol, -1)
                Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
                Chem.AssignAtomChiralTagsFromStructure(mol, -1)
                remained_smiles.append(smiles)
                canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
            except:
                print('can not convert this {} smiles'.format(smiles))
                del_smiles_list.append(smiles)
        print("number of successfully processed smiles: ", len(remained_smiles))
        return del_smiles_list


    def predict(self, predict_smiles, graph_dict=None):
        '''

        :param predict_smiles: a list of predict smiles
        :return:
        '''

        if graph_dict is None:
            graph_dict = graph(predict_smiles)
        fold = 5
        model_list = []
        predict_list = []
        import json
        with open('{}/{}/lr_weight.json'.format(self.model_path, self.task_name), 'r') as f:
            weight_dict = json.loads(f.read())
            coef_ = np.array(weight_dict['coef_'])
            intercept_ = np.array(weight_dict['intercept_'])
        for i in range(fold):
            model_list.append(torch.load('{}/{}/fold_{}.pt'.format(self.model_path, self.task_name, str(i), )))
            predict_list.append([])
        if len(model_list) != 5:
            raise FileNotFoundError('not enough model')
        eval_loader = DataLoader(graph_dataset(predict_smiles, graph_dict), self.batch_size,collate_fn=null_collate,
                                 shuffle=False, worker_init_fn=np.random.seed(SEED))
        for num, model in enumerate(model_list):
            model.to(self.device)
            model.eval()

            for b, (smiles, atom, bond, bond_index, mol_index, _) in enumerate(eval_loader):
                atom = atom.to(self.device)
                bond = bond.to(self.device)
                bond_index = bond_index.to(self.device)
                mol_index = mol_index.to(self.device)
                mol_prediction = model(atom, bond, bond_index, mol_index)
                predict_list[num].extend(mol_prediction.squeeze(dim=1).detach().cpu().numpy())

        if self.type == 'regression':
            predict_mean = np.array(predict_list).sum(axis=0) / fold
            predict_weighted = np.dot(np.array(predict_list).transpose(), coef_.transpose()) + intercept_
            if self.weighted == "mean":
                predict_result = predict_mean
            else:
                predict_result = predict_weighted
            # predict_result = dict(zip(predict_smiles, predict_result))
            predict_result = {"Value":round(float(predict_result),4),"Unit":self.label_class}
        else:
            predict_list = softmax(predict_list, dim=2)
            predict_mean = np.array(predict_list).sum(axis=0) / fold
            predict_weighted = np.dot(predict_list[:, :, 1].transpose(), coef_.transpose()) + intercept_
            predict_weighted = 1 / (np.exp(-predict_weighted) + 1)
            predict_weighted = np.concatenate((1 - predict_weighted, predict_weighted), axis=1)
            predict_mean = softmax(predict_mean,dim=1)
            if self.weighted == "mean":
                predict_result = predict_mean.squeeze()
            else:
                predict_result = predict_weighted.squeeze()
            # predict_result = dict(zip(predict_smiles,[self.label_class[i] for i in np.argmax(predict_result)]))
            result_index = np.argmax(predict_result)
            predict_result = {"Result":self.label_class[result_index],"Probability":predict_result[result_index]}
        return predict_result

    def evaluate(self, target, pred):
        metrics_dict_weighted = {}
        if self.type == 'regression':
            for metric in ['explained_variance_score', 'mean_absolute_error', 'mean_squared_error', 'r2_score',
                                  'median_absolute_error']:
                metrics_dict_weighted.update({metric: round(build_metrics_func(metric)(target, pred), 4)})
            print(self.task_name + 'metrics_dict_weighted :', metrics_dict_weighted)
        else:
            for metric in  ['accuracy_score',  'f1_score', 'precision_score', 'recall_score','roc_auc_score','matthews_corrcoef'
                                      #'average_precision_score',
                                      ]:
                metrics_dict_weighted.update({metric: round(build_metrics_func(metric)(target, pred), 4)})
            print(self.task_name + 'metrics_dict_mean :', metrics_dict_weighted)


def softmax(x, dim=1):
    x = np.array(x, dtype=float)
    x = np.exp(x - np.expand_dims(np.max(x, axis=dim), dim))
    x = x / np.expand_dims(np.sum(x, axis=dim), dim)
    return x


def build_metrics_func(metric_name, need_max=True):
    from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, \
        mean_squared_log_error, r2_score, median_absolute_error
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, \
        roc_auc_score, matthews_corrcoef
    func = locals()[metric_name]
    if metric_name in ['accuracy_score', 'f1_score', 'recall_score', 'precision_score', 'matthews_corrcoef']:
        if need_max:
            return lambda x, y: func(x, np.argmax(y, axis=1))
        else:
            return lambda x, y: func(x, np.round(y))

    if metric_name in ['average_precision_score', 'roc_auc_score', ]:
        if need_max:
            return lambda x, y: func(x, y[:, 1])
        else:
            return lambda x, y: func(x, y)

    return locals()[metric_name]


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--param',type=str,required=False)
    parser.add_argument('-t','--task',type=str,required=False,help="task name",default='all')
    parser.add_argument('-i','--input',type=str,required=False,help="abs path of input")
    parser.add_argument('-o','--output',type=str,required=False,help="name of output")
    args = parser.parse_args()
    if args.param:
        with open(args.param) as f:
            param = json.load(f)
        input_path = param['input']
        output_path = param['output']
        task = param['task']
    else:
        input_path = args.input
        output_path = args.output
        task = [args.task]
    with open("param_conf.json") as f:
        param_conf = json.load(f)

    if task == ["all"]:
        task_list = list(param_conf.keys())
    else:
        task_list = [i for i in task if i not in param_conf.keys()]
        if task_list:
            raise ValueError(f"{','.join(task_list)} is unsupported task type")
        task_list = task

    with open(input_path) as f:
        input_lines = [line.strip() for line in f.readlines()]
    del_smiles = Attentivefp.pre_data(input_lines)
    if del_smiles:
        with open("log.txt",'w') as f:
            f.write(f"can not processed these smiles: {','.join(del_smiles)}")
        input_lines = [s for s in input_lines if s not in del_smiles]

    import collections
    output_lines = collections.OrderedDict()
    graph_smiles = graph(input_lines)

    for input_line in input_lines:
        output_line = collections.OrderedDict()
        for task_name in task_list:
            afp = Attentivefp(task_name,param_conf[task_name])
            output_line.update({param_conf[task_name]["name"]:afp.predict([input_line],graph_smiles)})
        output_lines.update({input_line:output_line})

    with open(output_path,'w') as f:
        json.dump(output_lines,f,indent=4)
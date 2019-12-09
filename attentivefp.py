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
import gc
import sys

sys.setrecursionlimit(50000)
import pickle
import random

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.nn.Module.dump_patches = True
import copy
import pandas as pd

# then import my own modules
from timeit import default_timer as timer
from AttentiveFP.featurizing import graph_dict as graph
from AttentiveFP.AttentiveLayers import Fingerprint, graph_dataset, null_collate, Graph, Logger, time_to_str





SEED = 168
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True

from rdkit import Chem
# from rdkit.Chem import AllChem
from rdkit.Chem import QED
from rdkit.Chem import rdMolDescriptors, MolSurf
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import seaborn as sns
from utils import Param

sns.set()
from IPython.display import SVG, display
#import sascorer

class Attentivefp(object):
    def __init__(self, filename, **kwargs):
        self.batch_size = 50
        self.epochs = 200
        self.p_dropout = 0.2
        self.fingerprint_dim = 128
        self.weight_decay = 5  # also known as l2_regularization_lambda
        self.learning_rate = 3.5
        self.K = 2
        self.T = 2
        self.param = None
        self.data_df = None
        self.label_class = None
        self.need_gpu = True
        self.param = Param(filename,'data/tang')
        self.predict_path = 'best'
        self.weighted = 'mean'
        self.gpu = 'cpu'
        for key, value in kwargs.items():
            if hasattr(self,key):
                setattr(self,key,value)

        if self.gpu == 'gpu':
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
            # cuda_aviable = torch.cuda.is_available()
            # device = torch.device(0)

    @staticmethod
    def pre_data(smiles_list):
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

    @staticmethod
    def run_data(data_df,name):
        smiles_list = data_df.SMILES.values
        del_smiles_list = Attentivefp.pre_data(smiles_list)   #TODO: changed need debug
        data_df = data_df[~data_df.SMILES.isin(del_smiles_list)]
        smiles_list = data_df.SMILES.values
        label_list = data_df.label.values
        graph_dict = graph(smiles_list, label_list, name)
        test_df = data_df.sample(frac=0.1, random_state=SEED)
        test_smiles = test_df.SMILES.values
        training_df = data_df.drop(test_df.index)
        training_smiles = training_df.SMILES.values
        print('train smiles:{}  test smiles:{}'.format(len(training_smiles), len(test_smiles)))
        return training_smiles,test_smiles,graph_dict

    def train(self):
        from sklearn.model_selection import KFold
        data_df, label_class = self.param.get_data()
        training_smiles, _, graph_dict = Attentivefp.run_data(data_df,self.param.name)
        kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
        train_fold = []
        valid_fold = []
        for k, (train_idx, valid_idx) in enumerate(kfold.split(training_smiles)):
            train_fold.append(train_idx)
            valid_fold.append(valid_idx)
        pwd = os.getcwd()
        if not os.path.exists(pwd + '/saved_models/' + self.param.name):
            os.mkdir(pwd + '/saved_models/' + self.param.name)
        if not os.path.exists(pwd + '/' + 'runs/' + self.param.name + self.param.time):
            os.mkdir(pwd + '/' + 'runs/' + self.param.name + self.param.time)

        from tensorboardX import SummaryWriter
        writer = SummaryWriter('runs/' + self.param.name + self.param.time)
        log = Logger()
        #log.open(f'bio/{self.param.name}_{self.param.time}.txt')

        f = '{:^5} | {:^7.4f} | {:^7.4f} | {:^7.4f} | {:^7} \n'
        log.write('epoch | loss | train MSE |  valid MSE |  time \n')
        start = timer()

        log2 = Logger()
        #log2.open(f'bio/{self.param.name}_best_{self.param.time}.txt')
        f2 = '{:^5} | {:^5} | {:^7.4f} | {:^7.4f} \n'

        best_param = {}
        for fold_index in range(5):

            model = Fingerprint(self.param.output_units_num, self.fingerprint_dim, K=self.K, T=self.T, p_dropout=self.p_dropout)
            if self.need_gpu:
                model.cuda()
            # if param.multi_task:
            #     from utils import MultiLoss
            #     loss_function = MultiLoss()
            #     print('loss func : Myloss')
            if self.param.type == 'regression':
                print('loss func : MSE')
                loss_function = nn.MSELoss()
            else:
                loss_function = nn.CrossEntropyLoss()
                print('loss func : cross')
            optimizer = optim.Adam(model.parameters(), 10 ** -self.learning_rate, weight_decay=10 ** -self.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6,
                                                             min_lr=0.0001)
            best_param[fold_index] = {}
            best_param[fold_index]["train_epoch"] = 0
            best_param[fold_index]["valid_epoch"] = 0
            best_param[fold_index]["train_loss"] = 9e8
            best_param[fold_index]["valid_loss"] = 9e8

            for epoch in range(800):
                model.train()
                train_loader = DataLoader(graph_dataset(training_smiles[train_fold[fold_index]], graph_dict), self.batch_size,
                                          collate_fn=null_collate,
                                          num_workers=8, pin_memory=True, drop_last=True, shuffle=True,
                                          worker_init_fn=np.random.seed(SEED))
                losses = []
                for b, (smiles, atom, bond, bond_index, mol_index, label) in enumerate(train_loader):
                    atom = atom.cuda()
                    bond = bond.cuda()
                    bond_index = bond_index.cuda()
                    mol_index = mol_index.cuda()
                    label = label.cuda()

                    mol_prediction = model(atom, bond, bond_index, mol_index)
                    if self.param.type == 'regression':
                        label = label.view(-1, self.param.output_units_num)
                    else:
                        label = torch.squeeze(label.long())
                    loss = loss_function(mol_prediction, label)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    losses.append(loss.item())
                losses = np.mean(losses)
                model.eval()
                train_loss = self.val(training_smiles[train_fold[fold_index]], graph_dict, model)
                valid_loss = self.val(training_smiles[valid_fold[fold_index]], graph_dict, model)
                scheduler.step(valid_loss)

                timing = time_to_str((timer() - start), 'min')
                log.write(f.format(epoch, losses, train_loss, valid_loss, timing))
                writer.add_scalars('loss', {'fold_' + str(fold_index) + 'train': train_loss,
                                            'fold_' + str(fold_index) + 'valid': valid_loss}, epoch)

                if train_loss < best_param[fold_index]["train_loss"]:
                    best_param[fold_index]["train_epoch"] = epoch
                    best_param[fold_index]["train_loss"] = train_loss
                if valid_loss < best_param[fold_index]["valid_loss"]:
                    best_param[fold_index]["valid_epoch"] = epoch
                    best_param[fold_index]["valid_loss"] = valid_loss
                    torch.save(model, 'saved_models/{}/fold_{}_{}_{}.pt'.format(self.param.name, str(fold_index), self.param.time,
                                                                                epoch))

                if epoch % 50 == 0 and epoch > 50:
                    try:
                        os.remove('saved_models/{}/fold_{}_{}_{}.pt'.format(self.param.name, str(fold_index), self.param.time,
                                                                            str(i for i in range(epoch - 50 * 2, epoch - 50))))
                    except(FileNotFoundError):
                        pass

                if (epoch - best_param[fold_index]["train_epoch"] > 30) or (epoch - best_param[fold_index]["valid_epoch"] > 10):
                    model = torch.load('saved_models/{}/fold_{}_{}_{}.pt'.format(self.param.name, str(fold_index),
                                                                                 self.param.time,
                                                                                 best_param[fold_index]["valid_epoch"]))
                    torch.save(model,
                               'saved_models/{}/fold_{}_{}_best.pt'.format(self.param.name, str(fold_index), self.param.time))

                    break

            log2.write('fold | epoch | train_MSE | valid MSE \n')
            log2.write(
                f2.format(fold_index, best_param[fold_index]["valid_epoch"], best_param[fold_index]["train_loss"],
                          best_param[fold_index]["valid_loss"]))

        writer.close()
        print(self.param.name + 'succeed')

    def val(self, smiles_list, graph_dict, model):
        eval_loss_list = []
        eval_loader = DataLoader(graph_dataset(smiles_list, graph_dict), self.batch_size, collate_fn=null_collate, num_workers=8,
                                 pin_memory=True, shuffle=False, worker_init_fn=np.random.seed(SEED))
        for b, (smiles, atom, bond, bond_index, mol_index, label) in enumerate(eval_loader):
            atom = atom.cuda()
            bond = bond.cuda()
            bond_index = bond_index.cuda()
            mol_index = mol_index.cuda()
            label = label.cuda()
            # if self.param.normalization:
            #     label = (label - mean_list[0]) / std_list[0]

            input = model(atom, bond, bond_index, mol_index)

            # if param.multi_task:
            #     loss_ = MultiLoss()
            #     loss = loss_(input, label.view(-1, param.task_num))
            #
            # else:
            if self.param.type == 'regression':
                loss = F.l1_loss(input, label.view(-1, self.param.output_units_num), reduction='mean')
            else:
                loss = F.cross_entropy(input, label.squeeze().long(), reduction='mean')

            loss = loss.cpu().detach().numpy()
            eval_loss_list.extend([loss])
        loss = np.array(eval_loss_list).mean()
        return loss #if not self.param.normalization else np.array(eval_loss_list) * std_list[0]

    def evaluate(self):
        data_df, label_class = self.param.get_data()
        _ ,test_smiles, graph_dict = Attentivefp.run_data(data_df,self.param.name)
        fold = 5
        model_list = []
        predict_list = []
        label_list = []
        for i in range(5):
            for save_time in [
                              '2019112710', '2019112712', '2019112713', '2019112808', '2019112810', '2019112811',
                              '2019112813', '2019112814', '2019112815', '2019112816', '2019112817', '2019112818',
                              '2019112820','2019112821', '2019112900', '2019120506',
                              '2019120408',
                              ]:
                try:
                    model_list.append(
                        torch.load('saved_models/{}/fold_{}_{}_best.pt'.format(self.param.name, str(i), save_time)))
                    break
                except FileNotFoundError:
                    pass
            predict_list.append([])
            label_list.append([])
        if len(model_list) != 5:
            raise FileNotFoundError('not enough model')
        eval_loader = DataLoader(graph_dataset(test_smiles, graph_dict), self.batch_size, collate_fn=null_collate,
                                 num_workers=8,
                                 pin_memory=True, shuffle=False, worker_init_fn=np.random.seed(SEED))
        for num, model in enumerate(model_list):
            model.eval()

            for b, (smiles, atom, bond, bond_index, mol_index, label) in enumerate(eval_loader):
                atom = atom.cuda()
                bond = bond.cuda()
                bond_index = bond_index.cuda()
                mol_index = mol_index.cuda()
                label = label.cuda()
                mol_prediction = model(atom, bond, bond_index, mol_index)
                predict_list[num].extend(mol_prediction.squeeze(dim=1).detach().cpu().numpy())
                label_list[num].extend(label.squeeze(dim=1).detach().cpu().numpy())
            #         print(predict.list)

        label = np.array(label_list).sum(axis=0) / fold

        from sklearn.linear_model import Ridge, LogisticRegression

        if self.param.type == 'regression':
            predict_mean = np.array(predict_list).sum(axis=0) / fold
            metrics_dict_mean = {}
            for metric in self.param.metrics:
                metrics_dict_mean.update({metric: round(build_metrics_func(metric)(label, predict_mean), 4)})
            print(self.param.name + 'metrics_dict_mean :', metrics_dict_mean)
            metrics_dict_weighted = {}
            clf = Ridge(alpha=.3)
            clf.fit(np.array(predict_list).transpose(), label)
            predict_weighted = clf.predict(np.array(predict_list).transpose())
            for metric in self.param.metrics:
                metrics_dict_weighted.update({metric: round(build_metrics_func(metric)(label, predict_weighted), 4)})
            print(self.param.name + 'metrics_dict_weighted :', metrics_dict_weighted)

        elif self.param.type == 'classification':
            predict_list = softmax(predict_list,dim=2)
            predict_mean = np.array(predict_list).sum(axis=0) / fold
            metrics_dict_mean = {}
            for metric in self.param.metrics:
                metrics_dict_mean.update({metric: round(build_metrics_func(metric)(label, softmax(predict_mean)), 4)})
            print(self.param.name + 'metrics_dict_mean :', metrics_dict_mean)

            clf = LogisticRegression()
            # clf2 = Ridge(alpha=0.3)
            clf.fit(predict_list[:, :, 1].transpose(), label)
            # clf2.fit(np.array(predict_list)[:,:,0].transpose(), -label+1)
            y_score_weighted = clf.predict_proba(predict_list[:, :, 1].transpose()).squeeze()
            metrics_dict_weighted = {}
            for metric in self.param.metrics:
                metrics_dict_weighted.update(
                    {metric: round(build_metrics_func(metric)(label, y_score_weighted), 4)})
            print(self.param.name + 'metrics_dict_weighted :', metrics_dict_weighted)
            # writer = SummaryWriter('runs/' + self.param.name + self.param.time)
            # writer.add_pr_curve('pr_curve_' + self.param.name, label, y_score_weighted[:, 1])
            # writer.add_pr_curve('pr_curve_' + self.param.name, label, y_score_mean)
            # writer.add_pr_curve('pr_curve_' + self.param.name, label, y_score_weighted[:, 1], global_step=1)
            # writer.close()

        import json
        if not os.path.exists('best/'+self.param.name):
            os.mkdir('best/'+self.param.name)
        with open('best/'+self.param.name+'/lr_weight.json','w') as f:

            f.write(json.dumps({'coef_':clf.coef_.tolist(),'intercept_':clf.intercept_.tolist()},))
        for i, model in enumerate(model_list):
            model.cpu()
            torch.save(model,'best/{}/fold_{}.pt'.format(self.param.name, str(i),))

    def predict(self,predict_smiles):
        self.param.get_data()  #TODO : 待删除 修改utils中用json保存每个任务的type，labelclass等
        del_smiles = Attentivefp.pre_data(predict_smiles)
        predict_smiles = [smiles for smiles in predict_smiles if smiles not in del_smiles]
        graph_dict = graph(predict_smiles)
        fold = 5
        model_list = []
        predict_list = []
        import json
        with open('{}/{}/lr_weight.json'.format(self.predict_path,self.param.name), 'r') as f:
            weight_dict = json.loads(f.read())
            coef_ = np.array(weight_dict['coef_'])
            intercept_ = np.array(weight_dict['intercept_'])
        for i in range(fold):
            model_list.append(torch.load('{}/{}/fold_{}.pt'.format(self.predict_path,self.param.name, str(i),)))
            predict_list.append([])
        if len(model_list) != 5:
            raise FileNotFoundError('not enough model')
        print("loader")
        eval_loader = DataLoader(graph_dataset(predict_smiles, graph_dict), self.batch_size, collate_fn=null_collate,
                                # num_workers=8,
                                 pin_memory=True, shuffle=False, worker_init_fn=np.random.seed(SEED))
        for num, model in enumerate(model_list):
            if self.gpu == 'gpu':
                model.cuda()
            model.eval()

            for b, (smiles, atom, bond, bond_index, mol_index,_) in enumerate(eval_loader):
                if self.gpu == 'gpu':
                    atom = atom.cuda()
                    bond = bond.cuda()
                    bond_index = bond_index.cuda()
                    mol_index = mol_index.cuda()
                mol_prediction = model(atom, bond, bond_index, mol_index)
                if self.gpu == 'gpu':
                    predict_list[num].extend(mol_prediction.squeeze(dim=1).detach().cpu().numpy())
                else:
                    predict_list[num].extend(mol_prediction.squeeze(dim=1).detach().numpy())
        if self.param.type == 'regression':
            predict_mean = np.array(predict_list).sum(axis=0) / fold
            predict_weighted = np.dot(np.array(predict_list).transpose(),coef_.transpose()) + intercept_
            if self.weighted == "mean":  #TODO:labelclass
                print(predict_mean)
            else:
                print(predict_weighted)

        elif self.param.type == 'classification':
            predict_list = softmax(predict_list, dim=2)
            predict_mean = np.array(predict_list).sum(axis=0) / fold
            predict_weighted = np.dot(predict_list[:, :, 1].transpose(),coef_.transpose())+ intercept_
            predict_weighted = 1/(np.exp(-predict_weighted)+1)
            predict_weighted = np.concatenate((1-predict_weighted,predict_weighted),axis=1)
            if self.weighted == "mean":
                print(predict_mean)
            else:
                print(predict_weighted)

def softmax_(x, dim=1):
    x = np.array(x, dtype=float)
    x = x.swapaxes(dim, -1)
    m = x.shape
    x = np.reshape(x, (-1, np.size(x, -1)))
    x = np.exp(x - np.reshape(np.max(x, axis=1), (-1, 1)))
    x = x / np.reshape(np.sum(x, axis=1), (-1, 1))
    x = np.reshape(x, m)
    x = x.swapaxes(dim, -1)
    return x

def softmax(x, dim=1):
    x = np.array(x, dtype=float)
    x = np.exp(x - np.expand_dims(np.max(x, axis=dim), dim))
    x = x / np.expand_dims(np.sum(x, axis=dim), dim)
    return x


def build_metrics_func(metric_name,need_max=True):
    from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, \
        mean_squared_log_error, r2_score, median_absolute_error, r2_score
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, \
        roc_auc_score,matthews_corrcoef
    func = locals()[metric_name]
    if metric_name in ['accuracy_score','f1_score','recall_score','precision_score','matthews_corrcoef']:
        if need_max:
            return lambda x,y:func(x,np.argmax(y,axis=1))
        else:
            return lambda x,y:func(x,np.round(y))


    if metric_name in ['average_precision_score','roc_auc_score',]:
        if need_max:
            return lambda x,y:func(x,y[:,1])
        else:
            return lambda x,y:func(x,y)

    return locals()[metric_name]


# for _, _, file_list in os.walk('data/tang'):
#     for file in file_list:
#         filename = file.split('.')[0]
#         # if filename not in ['M_CYPPro_I']:
#         #     continue
#         if filename in ['test', 'bioinformatics2019']:
#             continue
#         model1 = Attentivefp(filename)
#         model1.predict()
#         # model1.evaluate()
#         print(filename)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--function', type=str, choices=['train','evaluate','predict'], required=False,default='predict')
    parser.add_argument('-n', '--name', type=str, required=True, help="task name")
    parser.add_argument('-g', '--gpu', type=str, choices=['gpu','cpu'],required=False, default='cpu')
    parser.add_argument('-w', '--weighted', type=str, choices=['mean','weighted'],required=False, default='mean')
    parser.add_argument('-s', '--smiles', type=str, required=False, )
    args = parser.parse_args()

    afp_model = Attentivefp(args.name,weighted=args.weighted,gpu=args.gpu)
    if args.smiles is None:
        getattr(afp_model,args.function)()
    else :
        getattr(afp_model, args.function)([args.smiles])

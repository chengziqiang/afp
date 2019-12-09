import pandas as pd
import numpy as np
import time
class Param(object):
    def __init__(self, task_name, relt_path='data/tang', normalization = False,multi_task=False):
        self.path = relt_path   #数据相对路径

        file_list = {}
        self.metrics_list = {'regression': ['explained_variance_score', 'mean_absolute_error', 'mean_squared_error', 'r2_score',
                                  'median_absolute_error'],
                   'classification': ['accuracy_score',  'f1_score', 'precision_score', 'recall_score','roc_auc_score','matthews_corrcoef'
                                      #'average_precision_score',
                                      ]}
        file_list.update(get_file(self.path))
        self.name = task_name
        self.filename = self.path+'/'+file_list[self.name]
        self.output_units_num = None
        self.normalization = normalization
        self.time = time.strftime("%Y%m%d%H")
        self.type = None
        self.metrics = None
        self.multi_task = multi_task
        self.task_num = 1


    def check(self):
        pass


    def get_data(self, ):
        df = read_data(self.filename)
        if self.multi_task:
            pass
            # features_list = [
            #                  # 'xi (CYP1A2)',
            #                  # 'xi (CYP2C9)',
            #     # 'xi (CYP2C19)','xi (CYP2D6)','xi (CYP3A4)',
            #                  'IInh'
            #                  ]
            # columns = df.columns.values
            # features = set(features_list)&set(columns)
            # tasks = list(features)
            # tasks.sort(key=features_list.index)
            # if len(tasks) > 0:
            #     self.task_num = len(tasks)
            #     data = df[tasks].copy()
            #     data.insert(0,'SMILES',df['SMILES'])
            #     data.loc[:,'IInh'] = df['IInh'].values
            #     label_class = [0,1]
            #     self.type = 'classification'
            #     # self.output_units_num = 1 * self.task_num
            #     self.output_units_num = 2 * self.task_num
            #     self.multi_task = True
            #     for i in tasks:
            #         data.loc[data.loc[:,i] >= 0.5,i] = 1
            #         data.loc[data.loc[:,i] < 0.5,i] = 0
            #         # data.loc[data.loc[:, i] > 0.8, i] = 1
            #         # data.loc[data.loc[:,i] < 0.2,i] = 0
            #         # data.drop(data[(data.loc[:, i] != 0) & (data.loc[:, i] != 1)].index,axis=0,inplace=True)

        else:
            if 'LogS' in list(df.columns.values):
                label_list = df['LogS'].values
            elif 'IInh' in list(df.columns.values):
                df.loc[df.loc[:, 'IInh'] >= 0.5, 'IInh'] = 1
                df.loc[df.loc[:, 'IInh'] < 0.5, 'IInh'] = 0
                # df.loc[df.loc[:, i] > 0.8, i] = 1
                # df.loc[df.loc[:,i] < 0.2,i] = 0
                # df.drop(df[(df.loc[:, i] != 0) & (df.loc[:, i] != 1)].index,axis=0,inplace=True)
                label_list = df['IInh'].values
            elif '%F' in list(df.columns.values):
                label_list = df['%F'].values
            elif 'End point.1' in list(df.columns.values):
                label_list = df['End point.1'].values
            elif 'End points' in list(df.columns.values):
                label_list = df['End points'].values
            elif 'End Points' in list(df.columns.values):
                label_list = df['End Points'].values
            elif 'Labels' in list(df.columns.values):
                label_list = df['Labels'].values
            elif 'Label' in list(df.columns.values):
                label_list = df['Label'].values
            elif 'Endpoints' in list(df.columns.values):
                label_list = df['Endpoints'].values
            elif 'CLASS' in list(df.columns.values):
                label_list = df['CLASS'].values
            elif 'End Points (pIGCC50,ug/L )' in list(df.columns.values):
                label_list = df['End Points (pIGCC50,ug/L )'].values
            elif 'LogPapp(cm/s)' in list(df.columns.values):
                label_list = df['LogPapp(cm/s)'].values
            elif 'End Points (pLC50,mg/L )' in list(df.columns.values):
                label_list = df['End Points (pLC50,mg/L )'].values
            elif 'labels' in list(df.columns.values):
                label_list = df['labels'].values
            else:
                raise ValueError(self.name + 'can not find feature')
            smiles = list(df['SMILES'].values)
            label_class = list(set(label_list))

            if len(label_class) != 2:
                # raise ValueError('classification task only support binary')
                # print('labelclass', label_class[:2])
                label_class = None
                label = np.array(label_list)
                self.type = 'regression'
                self.output_units_num = 1 * self.task_num

            else:
                label = np.zeros(len(label_list), dtype=int)
                for n, i in enumerate(label_list):
                    label[n] = int(label_class.index(i))
                self.type = 'classification'
                self.output_units_num = len(label_class) * self.task_num
            data = pd.DataFrame(smiles, columns=['SMILES'])
            data['label'] = label
            print('{} :{}  smiles num :{} success'.format(self.name, self.type, len(smiles)))
            tasks = ['label']
        # self.metrics = self.metrics_list['classification']
            self.metrics = self.metrics_list[self.type]
            return data, label_class
import torch
from torch import nn
from torch.nn import functional as F
class MultiLoss(nn.Module):
    def __init__(self):
        super(MultiLoss,self).__init__()
        self.reduction = 'none'
    def forward(self, predict,target,reduction='none'):
        if not (target.size() == predict.size()):
            print("Using a target size ({}) that is different to the input size ({}). "
                          "This will likely lead to incorrect results due to broadcasting. "
                          "Please ensure they have the same size.".format(target.size(), predict.size()),)
        self.reduction = reduction
        loss_mse = F.mse_loss(predict,target,reduction=self.reduction)
        zeros = torch.zeros_like(predict).cuda()
        ones = torch.ones_like(predict).cuda()
        loss_weight = torch.where(target==0,zeros,ones).cuda()
        loss = torch.mean(loss_mse.mul(loss_weight))

        # loss_mse.register_hook(save_grad('loss_mse'))
        # print('loss_w',loss_weight)
        return loss

def get_file(path):
    import os
    file_dict = {}
    for _,_,file_list in os.walk(path):
        for file in file_list:
            filename = file.split('.')[0]
            file_dict.update({filename:file})
    return file_dict


def save_grad(name):
    def hook(grad):
        print('grad',grad)
    return hook

def read_data(filename):
    if filename.split('.')[-1] == 'xls':
        df = pd.read_excel(filename)

    elif filename.split('.')[-1] == 'txt':
        try:
            df = pd.read_table(filename)
        except:
            df = pd.read_table(filename,encoding='gb18030')

    elif filename.split('.')[-1] == 'csv':
        df = pd.read_csv(filename)

    else:
        raise ValueError('unsupported file type ')

    print('{} :{}'.format(filename, df.columns.values))
    return df

if __name__ == '__main__':
    import os
    # for _,_,file_list in os.walk('data/tang'):
    #     print(file_list)
        # for file in file_list:
        #     filename = file.split('.')[0]
        #     if filename != 'M_CYPPro_I' :
        #         param = Param(filename)
        #         param.get_data()


    a ='M_CYPPro_I'
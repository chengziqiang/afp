# Introduction
---
## Requirements
* python3 
* pytorch
* cuda
* pytorch-geomtric: https://github.com/rusty1s/pytorch_geometric
* rdkit: http://www.rdkit.org/docs/Install.html
## Start 
* run example 
 `$ cd afp`

 `$ python attentivefp.py -n "M_CYPPro_I" -s "COC1=CC=C(C=C1)C2=CC(=NC(=N2)N3N=C(C)C=C3C)C(F)F"`
* -f 可选'train','evaluate','predict', 默认'predict'
* -n 预测任务名称
* -w 可选'mean','weighted', 使用模型平均或加权预测, 默认"weighted"
* -s 需要预测的分子smiles式

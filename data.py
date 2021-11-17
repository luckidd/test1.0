#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :data.py
# @Time      :2021/11/17 23:30
# @Author    :Wang Jianhang

from sklearn.datasets import load_iris

iris_dataset = load_iris()
# load_iris返回的iris对象是一个Bunch对象，与字典非常相似，里面包含键和值
# print("Keys of iris_dataset:\n{}".format(iris_dataset.keys()))
print("Keys of iris_dataset:\n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:139] + "\n...")
print("Target names:{}".format(iris_dataset['target_names']))
print("Feature names:\n{}".format(iris_dataset['feature_names']))
print("Type of data:{}".format(type(iris_dataset['data'])))
print("Shape of data:{}".format(iris_dataset['data'].shape))
# Shape of data:(150, 4) 样本*特征数
print("First five rows of data:\n{}".format(iris_dataset['data'][:5]))
print("Type of target:{}".format(type(iris_dataset['target'])))
print("Shape of target:{}".format(iris_dataset['target'].shape))
print("Targt:\n{}".format(iris_dataset['target']))

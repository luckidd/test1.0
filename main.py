#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :data.py
# @Time      :2021/11/17 23:30
# @Author    :Wang Jianhang

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.neighbors import KNeighborsClassifier

# todo:分类数据
iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
print("X_train shape:{}".format(X_train.shape))
print("y_train shape:{}".format(y_train.shape))
print("X_test shape:{}".format(X_test.shape))
print("y_test shape:{}".format(y_test.shape))

# todo:观察数据

# 利用X_train中的数据创建DataFrame
# 利用iris_dataset.feature_names中的字符串对数据进行标记
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# columns=iris_dataset.feature_names 给列做标记
# 利用DataFrame创建散点图矩阵，按y_train着色
# grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=0.8, cmap=mglearn.cm3)
# todo:#↑

# 构建模型：构建第一个模型：k近邻算法
# 考虑训练集中与新数据点最近的任意k个邻居（比如说，距离最近的3个或5个邻居），而不是只考虑最近的那一个。然后，我们可以用这些邻居中数量做多的类别做出预测。
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
X_new = np.array([[5, 2.9, 1, 0.2]])
# 因为scikit-learn的输入数据必须是二维数组
print("X_new.shape:{}".format(X_new.shape))
prediction = knn.predict(X_new)
print("Prediction:{}".format(prediction))
print("Predicted target name:{}".format(iris_dataset['target_names'][prediction]))

# todo:评估模型
y_pred = knn.predict(X_test)
print("Test set predictions:{}".format(y_pred))
print("Test set score:{:.2f}".format(np.mean(y_pred == y_test)))
print("Test set sore:{:.2f}".format(knn.score(X_test, y_test)))

#!~/anaconda3/bin/python3.9
# -*- coding: UTF-8 -*- #
"""
@Author     : Administrator
@Time       : 2022-07-07 13:54 
@Environment: PyCharm
@Version    : v.1.0. Train test split for multi-class data, with random seed.
"""
import copy
import numpy as np
import pandas as pd


def split_ykp(index_list, test_size, random_seed=1):
    # 划分训练集和测试集索引
    """
    Input:
        index_list       —— List需要划分的样本索引
        test_size        —— Float训练测试划分比例,即测试集的比例,比如为0.2
        random_seed      —— Int随机种子,默认为1
    Output:
        train_index      —— List训练集索引
        test_index       —— List测试集索引
    """
    if np.floor(len(index_list)*(1-test_size)) == 0:
        index_train = index_list[0:1]
        index_test = index_list[0:]
    else:
        import random
        random.seed(random_seed)
        num_train = int(np.floor(len(index_list)*(1-test_size)))
        index_train = random.sample(index_list, num_train)
        index_test = [i for i in index_list if i not in index_train]
    return index_train, index_test


def train_test_split_multi_class_ykp(data_x, data_y, test_size=0.2, random_seed=1):
    # 假设标签类别包含4类,即0-1-2-3,,划分后的样本要保证训练集和测试集中的每类标签都有, 且按照比例划分
    """
    Input:
        data_x           —— DataFrame 样本特征
        data_y           —— Series 样本标签
        test_size        —— Float 训练测试划分比例,即测试集的比例,比如为0.2
    Output:
        data_train_x          —— DataFrame 训练集特征
        data_test_x           —— DataFrame 测试集特征
        data_train_y          —— Series 训练集标签
        data_test_y           —— Series 测试集标签
    """
    sel_features = list(data_x.columns)  # 取出所有特征
    class_num = len(data_y.unique())  # 类别数量
    temp_df = copy.deepcopy(data_x)  # 将data_x复制一份,用于划分训练集和测试集
    temp_df['label'] = data_y  # 将标签加入到temp_df中
    train_index_all = []  # 存储所有训练集索引
    test_index_all = []  # 存储所有测试集索引
    for i in range(class_num):
        all_index_i = temp_df[temp_df['label'] == i].index.tolist()  # 取出第i类标签的所有索引
        train_test_index = split_ykp(all_index_i, test_size, random_seed=random_seed)  # 划分训练集和测试集索引
        train_index_all += train_test_index[0]  # 将训练集索引加入到train_index_all中
        test_index_all += train_test_index[1]  # 将测试集索引加入到test_index_all中
    data_train_x = copy.deepcopy(temp_df.loc[train_index_all, sel_features])  # 取出训练集特征
    data_test_x = copy.deepcopy(temp_df.loc[test_index_all, sel_features])  # 取出测试集特征
    data_train_y = pd.Series(copy.deepcopy(temp_df.loc[train_index_all, ['label']].iloc[:, -1]))  # 取出训练集标签
    data_test_y = pd.Series(copy.deepcopy(temp_df.loc[test_index_all, ['label']].iloc[:, -1]))  # 取出测试集标签
    return data_train_x, data_test_x, data_train_y, data_test_y


if __name__ == '__main__':
    import os
    path = os.getcwd() + "\\"
    # 设置数据文件名称
    data_file = "Wine.3C.xlsx"
    sheet_name = "Wine.3C"
    # load data
    data = pd.read_excel(path + data_file, sheet_name=sheet_name)
    # split data
    data_x = data.iloc[:, :-1]
    data_y = data.iloc[:, -1]
    data_train_x, data_test_x, data_train_y, data_test_y = train_test_split_multi_class_ykp(
        data_x, data_y, test_size=0.4, random_seed=1)
    # print result
    print("All:   ", data_x.shape, "\t", np.bincount(data_y)/len(data_y))
    print("Train: ", data_train_x.shape, "\t", np.bincount(data_train_y)/len(data_train_y))
    print("Test:  ", data_test_x.shape, "\t", np.bincount(data_test_y)/len(data_test_y))
    # save data into excel
    writer = pd.ExcelWriter(path + "Wine.3C.train_test_split.xlsx")
    data_train_x.to_excel(writer, sheet_name="train_x")
    data_test_x.to_excel(writer, sheet_name="test_x")
    data_train_y.to_excel(writer, sheet_name="train_y")
    data_test_y.to_excel(writer, sheet_name="test_y")
    writer.save()

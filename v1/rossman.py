#!/usr/bin/env python
# -*- coding: gbk -*-
"""
File: rossman.py
Author: wufan0920@163.com
Date: 2017/01/03 15:03:23
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import datetime
import sys
import os

def preprocess_data(data):
    #清除0,'0'冗余
    data.loc[data.StateHoliday == 0, 'StateHoliday'] = '0'
    
    #处理categorical feature
    week = pd.get_dummies(data.DayOfWeek, prefix = 'week')
    holiday = pd.get_dummies(data.StateHoliday, prefix = 'holiday')
    stype = pd.get_dummies(data.stype, prefix = 'stype')
    assort = pd.get_dummies(data.assort, prefix = 'assort')

    data = pd.concat([data, week, holiday, stype, assort], axis = 1)
    data.drop(['DayOfWeek', 'StateHoliday', 'stype', 'assort'], axis = 1, inplace = True)
    return data

def feature_engineering(data, store):
    lines = data.shape[0]
    month_mapping = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    #为训练/测试数据加上storetype assortment promo2 have_competition competition_distance列
    extra_columns = {'promo2': np.array([0] * lines, dtype = 'int64'),
            'have_competition': np.array([0] * lines, dtype = 'int64'),
            'competition_distance': np.array([0] * lines, dtype = 'float64'),
            'stype': np.array(['a'] * lines, dtype = 'object'),
            'assort': np.array(['a'] * lines, dtype = 'object')}
    extra_columns = pd.DataFrame(extra_columns)
    data = pd.concat([data, extra_columns], axis = 1)

    #遍历数据查找store表中对应列处理后填充
    #TODO pandas有个merge函数.... 可以先合并再特征处理
    for index in range(lines):
        store_id = data.at[index, 'Store']
        date = data.at[index, 'Date']
        date = datetime.datetime.strptime(date, "%Y-%m-%d")
        month = date.timetuple().tm_mon
        year = date.timetuple().tm_year
        week = date.isocalendar()[1]

        #stype assort直接复制
        data.at[index, 'stype'] = store[store.Store == store_id].StoreType.iat[0]
        data.at[index, 'assort'] = store[store.Store == store_id].Assortment.iat[0]

        #compete相关数据
        compete_month = store[store.Store == store_id].CompetitionOpenSinceMonth.iat[0]
        compete_year = store[store.Store == store_id].CompetitionOpenSinceYear.iat[0]
        if compete_year < year or (compete_year == year and compete_month <= month):
            data.at[index, 'have_competition'] = 1
            data.at[index, 'competition_distance'] = store[store.Store == store_id].CompetitionDistance.iat[0]

        #promo2相关数据
        if store[store.Store == store_id].Promo2.iat[0] == 0:
            data.at[index, 'promo2'] = 0
        else:
            promo_week = store[store.Store == store_id].Promo2SinceWeek.iat[0]
            promo_year = store[store.Store == store_id].Promo2SinceYear.iat[0]
            promo_interval = store[store.Store == store_id].PromoInterval.iat[0]
            if promo_year < year or (promo_year == year and promo_week <= week):
                if month_mapping[month] in promo_interval:
                    data.at[index, 'promo2'] = 1

    return data

def generate_data(path):
    train_path = path + '/' + 'train.csv'
    test_path = path + '/' + 'test.csv'
    store_path = path + '/' + 'store.csv'

    store = pd.read_csv(store_path)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    #扩展训练测试数据特征
    train = feature_engineering(train, store)
    test = feature_engineering(test, store)
    train.to_csv("modify_train.csv")
    test.to_csv("modify_test.csv")

    #数据预处理
    test = preprocess_data(test)
    train = preprocess_data(train)

    return train, test

#均方比例误差
def rmspe(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", np.sqrt(np.mean(((y - yhat)/y) ** 2))

def training(data):
    #处理训练数据
    print 'training start'

    #close的数据不参与训练
    data.drop(data[data.Open == 0].index, inplace = True)

    #仅使用销量大于0的数据训练
    data = data[data.Sales > 0]

    #Store Customers Open Date 不做特征
    data.drop(['Store', 'Customers', 'Open', 'Date'], axis = 1, inplace = True)
    data.drop(['holiday_b', 'holiday_c'], axis = 1, inplace = True)
    train, valid = train_test_split(data, test_size=0.01, random_state=10)

    #train_labels = train.Sales
    train_labels = np.log1p(train.Sales)
    train.drop(['Sales'], axis = 1, inplace = True)

    #valid_labels = valid.Sales
    valid_labels = np.log1p(valid.Sales)
    valid.drop(['Sales'], axis = 1, inplace = True)

    param = {"objective": "reg:linear",
            "booster": "gbtree",
            "eta": 0.1, 
            "max_depth": 10,
            "lambda": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.4,
            "min_child_weight": 6,
            "silent": 1}
    num_rounds = 1200

    xgtrain = xgb.DMatrix(train, label = train_labels)  
    xgvalid = xgb.DMatrix(valid, label = valid_labels)  
  
    watchlist = [(xgtrain, 'train'), (xgvalid, 'valid')]  
    model = xgb.train(list(param.items()), xgtrain, num_rounds, watchlist, early_stopping_rounds=200,
            feval=rmspe, verbose_eval=True)
    return model

def predicting(model, data):
    print 'predicting start'
    close_list = data[data.Open == 0].index
    #Store Customers Open 不做特征
    data.drop(['Store', 'Open', 'Date', 'Id'], axis = 1, inplace = True)
    xgtest = xgb.DMatrix(data)

    preds = model.predict(xgtest)
    preds = np.expm1(preds)
    #preds[close_list] = 0

    np.savetxt('rossmann.csv', np.c_[range(1, len(data) + 1), preds], delimiter=',',  
            header='Id,Sales', comments='', fmt='%d')  

if __name__=='__main__':
    train_set, test_set = generate_data('data')
    model = training(train_set)
    model.save_model('rossmann.model')
    predicting(model, test_set)


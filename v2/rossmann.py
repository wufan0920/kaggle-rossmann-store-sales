#!/usr/bin/env python
# -*- coding: gbk -*-
"""
File: rossmann.py
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
    stype = pd.get_dummies(data.StoreType, prefix = 'stype')
    assort = pd.get_dummies(data.Assortment, prefix = 'assort')

    data = pd.concat([data, week, assort, stype, holiday], axis = 1)
    drop_list = ['DayOfWeek', 'StateHoliday', 'StoreType', 'Assortment',
            'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
            'Promo2SinceWeek', 'Promo2SinceYear', 'Promo2', 'PromoInterval',
            'year', 'month', 'week', 'month_abbr', 'datetime']
    data.drop(drop_list, axis = 1, inplace = True)
    return data

def feature_engineering(data):
    month_mapping = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

    data['datetime'] = data.Date.map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
    data['year'] = data.datetime.map(lambda x: x.timetuple().tm_year)
    data['month'] = data.datetime.map(lambda x: x.timetuple().tm_mon)
    data['week'] = data.datetime.map(lambda x: x.isocalendar()[1])

    #时间相关数据
    #竞争对手已开张持续时间
    data['competition_last_month'] = 12 * (data.year - data.CompetitionOpenSinceYear) + \
            (data.month - data.CompetitionOpenSinceMonth)
    data['competition_last_month'] = data.competition_last_month.apply(lambda x: x if x > 0 else 0)

    #持续促销已开始时间
    data['promo2_last_month'] = 12 * (data.year - data.Promo2SinceYear) + \
            (data.week - data.Promo2SinceWeek) / 4.0
    data['promo2_last_month'] = data.promo2_last_month.apply(lambda x: x if x > 0 else 0)
    data.loc[data.Promo2 == 0, 'promo2_last_month'] = 0
    #是否在促销期间
    data['month_abbr'] =  data.month.map(month_mapping)
    data['in_promo2'] = data.apply(lambda x: 1 if x.month_abbr in x.PromoInterval else 0,
            axis = 1)
    data.loc[data.promo2_last_month == 0, 'in_promo2'] = 0

    return data

def generate_data(path):
    train_path = path + '/' + 'train.csv'
    test_path = path + '/' + 'test.csv'
    store_path = path + '/' + 'store.csv'

    store = pd.read_csv(store_path)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    #store的空值处理
    store.loc[store.CompetitionOpenSinceMonth.isnull(), 'CompetitionOpenSinceMonth'] = \
            store.CompetitionOpenSinceMonth.mean()
    store.loc[store.CompetitionOpenSinceYear.isnull(), 'CompetitionOpenSinceYear'] = \
            store.CompetitionOpenSinceYear.mean()
    store.loc[store.CompetitionDistance.isnull(), 'CompetitionDistance'] = \
            store.CompetitionDistance.mean()
    store.loc[store.PromoInterval.isnull(), 'PromoInterval'] = '' 


    #扩展训练测试数据特征
    train = pd.merge(train, store, on = 'Store')
    test = pd.merge(test, store, on = 'Store')

    train = feature_engineering(train)
    test = feature_engineering(test)

    #数据预处理
    train = preprocess_data(train)
    test = preprocess_data(test)

    return train, test

#均方比例误差
def rmspe(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", np.sqrt(np.mean(((y - yhat)/y) ** 2))

def training(data):
    print 'training start'

    #close的数据不参与训练
    data = data[data.Open != 0]

    #仅使用销量大于0的数据训练
    data = data[data.Sales > 0]

    #Customers Open Date 不做特征
    data.drop(['Customers', 'Open', 'Date'], axis = 1, inplace = True)
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
    lines = data.shape[0]
    close_list = data[data.Open == 0].index
    index = data.Id
    #Customers Open 不做特征
    data.drop(['Open', 'Date', 'Id'], axis = 1, inplace = True)
    #测试集合不含两类假日
    data['holiday_b'] = 0
    data['holiday_c'] = 0
    xgtest = xgb.DMatrix(data)

    preds = model.predict(xgtest)
    preds = np.expm1(preds)
    #preds[close_list] = 0

    #np.savetxt('rossmann.csv', np.c_[range(1, len(data) + 1), preds], delimiter=',',  
    #        header='Id,Sales', comments='', fmt='%d')  
    np.savetxt('rossmann.csv', np.c_[index, preds], delimiter=',',  
            header='Id,Sales', comments='', fmt='%d')  

if __name__=='__main__':
    train_set, test_set = generate_data('data')
    train_set.to_csv("modified_train.csv")
    test_set.to_csv("modified_test.csv")
    model = training(train_set)
    model.save_model('rossmann.model')
    predicting(model, test_set)


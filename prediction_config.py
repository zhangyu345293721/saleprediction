# -*- coding:utf-8 -*-
'''
author:zhangyu
date:2021/2/24
description:算法配置文件
'''
from datetime import datetime

start_pred_dt = datetime(2020, 12, 13)  # 开始预测的时间
gap = 0  # 实际预测时间与开始预测时间的间隔
periods = 4  # 预测周期，即预测次数

MAX_ROUNDS = 700
params = {
    "num_leaves": 80,
    "objective": "regression",
    "min_data_in_leaf": 200,
    "learning_rate": 0.01,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "metric": "l2",
    "num_threads": 4,
}

EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 50

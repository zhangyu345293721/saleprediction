# -*- coding:utf-8 -*-
'''
author:zhangyu
date:2021/2/24
description:进行特征工程
'''

import gc
import numpy as np
import pandas as pd
from typing import Union, Tuple, List
from datetime import datetime, timedelta


def get_pre_vals(df: pd.DataFrame, dt: datetime, periods: int = 3, freq: str = "W",
                 need_index: bool = False, ) -> pd.DataFrame:
    '''
        获取预处理值
    Args:
        df: df
        dt: 时间
        periods: 时期
        freq: 频率
        need_index: 需要下标
    Returns:
        pd
    '''
    if freq == "H":
        end_dt = dt - timedelta(hours=1)
    elif freq == "D":
        end_dt = dt - timedelta(days=1)
    elif freq == "W":
        end_dt = dt - timedelta(weeks=1)
    elif freq == "M":
        end_dt = dt - timedelta(weeks=4)
    else:
        raise Exception("[ERROR] The frequency is illegal!!!")
    res = df[pd.date_range(end=end_dt, periods=periods, freq=freq)]
    return res if need_index else res.reset_index(drop=True)


def get_val(df: pd.DataFrame, dt: datetime, freq: str = "W", need_index: bool = False) -> Union[np.ndarray, pd.Series]:
    '''
        获取变量
    Args:
        df: df
        dt: 时间
        freq: 频率
        need_index: 需要下标
    Returns:
        union
    '''
    res = df[pd.date_range(start=dt, periods=1, freq=freq)].iloc[:, 0]
    return res if need_index else res.values


def prepare_dataset(df: pd.DataFrame, start_pred_dt: datetime, gap: int = 0, add_cate_feat: bool = True,
                    name_prefix: str = None, is_train: bool = True, ) -> Union[
    pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
    '''
        准备数据集
    Args:
        df: pd
        start_pred_dt: 开始时间
        gap: 间距
        add_cate_feat: 增加特征
        name_prefix:名字前缀
        is_train:是否训练
    Returns:
        Union
    '''
    X = {}
    for i in [3, 7, 14, 21]:
        tmp = get_pre_vals(df, start_pred_dt, periods=i)  # 前i个值
        X["mean_pre_%s" % i] = tmp.mean(axis=1).values  # 前i个值的平均值
        X["median_pre_%s" % i] = tmp.median(axis=1).values  # 前i个值的中位数
        X["max_pre_%s" % i] = tmp.max(axis=1).values  # 前i个值的最大值
        X["min_pre_%s" % i] = tmp.min(axis=1).values  # 前i个值的最小值
        # X['std_pre_%s' % i] = tmp.std(axis=1).values  # 前i个值的标准差
        # X['diff_mean_pre_%s' % i] = tmp.diff(axis=1).mean(axis=1).values  # 前i个值的平均一阶差分

    for i in [3, 6, 9, 12]:
        tmp = get_pre_vals(df, start_pred_dt - timedelta(weeks=4), periods=i)
        X["mean_last_month_pre_%s" % i] = tmp.mean(axis=1).values
        X["median_last_month_pre_%s" % i] = tmp.median(axis=1).values
        X["max_last_month_pre_%s" % i] = tmp.max(axis=1).values
        X["min_last_month_pre_%s" % i] = tmp.min(axis=1).values

    for i in [3, 7, 14, 21]:
        tmp = get_pre_vals(df, start_pred_dt, periods=i)  # 前i个值
        X["has_sale_weeks_pre_%s" % i] = (tmp > 0).sum(axis=1).values  # 前i个值有销量的数目
        X["last_has_sale_week_pre_%s" % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values  # 前i个值最后一次有销量到这周的周数
        X["first_has_sale_week_pre_%s" % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values  # 前i个值第一次有销量到这周的周数

    # 前i个值
    for i in range(1, 15):  # range(1, 8) or range(1, 15)
        tmp_dt = start_pred_dt - timedelta(weeks=i)
        X["pre_%s" % i] = get_val(df, tmp_dt)

    # 历史同期的取值
    for i in range(1, 3):
        tmp_dt = start_pred_dt - timedelta(weeks=52 * i)
        X["history_pre_%s" % i] = get_val(df, tmp_dt)

    X = pd.DataFrame(X)

    if add_cate_feat:
        X["pred_day"] = start_pred_dt.day
        X["pred_weekday"] = start_pred_dt.weekday()

    if name_prefix is not None:
        X.columns = ["%s_%s" % (name_prefix, c) for c in X.columns]

    if is_train:
        y = get_val(df, start_pred_dt)
        return X, y
    else:
        return X


def prepare_training_set(df: pd.DataFrame, item_info: pd.DataFrame, dts: List[datetime], gap: int) -> Union[
    pd.DataFrame, pd.DataFrame]:
    '''
        准备集合
    Args:
        df: df
        item_info: 商品信息
        dts: 时间表
        gap: 间距
    Returns:
        union
    '''
    X_list, y_list = [], []
    for dt in dts:
        X_tmp, y_tmp = prepare_dataset(df, dt, gap)
        X_tmp = pd.concat([X_tmp, item_info.reset_index(drop=True)], axis=1)
        X_list.append(X_tmp)
        y_list.append(y_tmp)
        del X_tmp, y_tmp
        gc.collect()

    X_train = pd.concat(X_list, axis=0)
    y_train = np.concatenate(y_list, axis=0)
    return X_train, y_train


def prepare_val_set(df: pd.DataFrame, item_info: pd.DataFrame, dt: datetime, gap: int) -> Union[
    pd.DataFrame, pd.DataFrame]:
    '''
        准备集合
    Args:
        df: df
        item_info: 商品info
        dt: datetime
        gap: int
    Returns:
        union
    '''
    X_val, y_val = prepare_dataset(df, dt, gap)
    X_val = pd.concat([X_val, item_info.reset_index(drop=True)], axis=1)
    return X_val, y_val


def get_pre_dts(dt: datetime, periods: int = 3) -> List[datetime]:
    '''
        获取时间链表
    Args:
        dt: 时间
        periods: 时期
    Returns:
        时间链表
    '''
    dts = []
    for i in range(1, periods + 1):
        dts.append(dt - timedelta(weeks=i))
    return dts


def prepare_testing_set(df: pd.DataFrame, item_info: pd.DataFrame, dt: datetime, gap: int) -> pd.DataFrame:
    '''
        整备测试集
    Args:
        df: df
        item_info: 商品信息
        dt: datetime
        gap: 间距
    Returns:
       测试数据集合
    '''
    X_test = prepare_dataset(df, dt, gap, is_train=False)
    X_test = pd.concat([X_test, item_info.reset_index(drop=True)], axis=1)
    return X_test


def prepare_training_dts(dt: datetime, gap: int, rounds=3 * 7):
    '''
        获取开始预测datetime
    Args:
        dt: 时间
        gap: 间距
        rounds: 次数
    Returns:
        时间链表
    '''
    upper_dt = dt - timedelta(weeks=gap)
    return get_pre_dts(upper_dt, rounds)

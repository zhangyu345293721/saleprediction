# -*- coding:utf-8 -*-
'''
author:zhangyu
date:2021/2/24
description:数据加载和数据预处理
'''
from pandas import DatetimeIndex
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import gc
import numpy as np
import datetime


def get_date_range() -> DatetimeIndex:
    '''
          生成时间序列
        - 2018年第一周是2017.12.31开始
        - 2019年的第一周是2018.12.30开始
        - 2020年的第一周是2020.1.5开始
        - 2021年的第一周是2021.1.3开始
    Returns:
       时间序列
    '''
    date1 = pd.date_range(
        start="2017.12.31",
        periods=53,
        freq="W",
    )[1:]
    date2 = pd.date_range(
        start="2018.12.30",
        periods=53,
        freq="W",
    )[1:]
    date3 = pd.date_range(
        start="2020.1.5",
        periods=53,
        freq="W",
    )[1:]
    return date1.append(date2).append(date3)


def get_sale_per_week() -> pd.DataFrame:
    '''
        读取csv文件的信息，并将信息进行转换
    Returns:
        df:sale_per_day
    '''
    df = pd.read_csv("generate_data.csv", parse_dates=[1])
    df = df[['0CM_CDT5', '0FISCPER', 'ZPI_SVL']]
    df.columns = ["sku_id", "sale_date", "sale_qty"]
    df["sku_id"] = df.sku_id.astype(str)
    tmp_df = df.groupby(by=["sku_id", "sale_date"])[["sale_qty"]].max()
    sale_per_week = tmp_df.unstack()
    sale_per_week.columns = get_date_range()
    date_time1 = datetime.datetime(2017, 12, 31)
    sale_per_week[date_time1] = np.nan
    date_time2 = datetime.datetime(2018, 12, 30)
    sale_per_week[date_time2] = np.nan
    date_time3 = datetime.datetime(2020, 1, 5)
    sale_per_week[date_time3] = np.nan
    del tmp_df
    gc.collect()
    return sale_per_week


def get_item_info(sale_per_week: pd.DataFrame) -> pd.DataFrame:
    '''
        更加item信息重新进行编码
    Args:
        sale_per_week: 销量数据
    Returns:
        商品编码后的信息
    '''
    item_info = pd.DataFrame(sale_per_week.index)
    encoder = LabelEncoder()
    item_info["sku_id"] = encoder.fit_transform(item_info["sku_id"])
    return item_info


if __name__ == '__main__':
    sale_per_week = get_sale_per_week()
    # get_item_info(sale_per_week)
    print(get_date_range())
    print(np.nan)
    print()
    d2 = datetime.datetime(2018, 12, 30)
    d1 = pd.date_range(
        start="2018.12.30",
        periods=1,
        freq="W",
    )[0]
    print(d2 == d1)

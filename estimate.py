# -*- coding:utf-8 -*-
'''
author:zhangyu
date:2021/2/24
description:结果评估
'''
from datetime import timedelta
from typing import List

from load_data import *
from prediction_config import *


def estimate(ys_test: List[int], preds_test: List[int]) -> pd.DataFrame:
    '''
        预测值和真实值合并
    Args:
        ys_test: 测试真是值
        preds_test:预测值
    Returns:
        df
    '''
    sale_per_week = get_sale_per_week()
    columns = [start_pred_dt + timedelta(weeks=i + gap) for i in range(periods)]
    df_test = pd.DataFrame(np.array(ys_test).transpose(), index=sale_per_week.index, columns=columns)
    df_test = df_test.stack().to_frame("y_true")
    df_test.index.names = ["sku_id", "sale_date"]
    df_pred = pd.DataFrame(
        np.array(preds_test).transpose(), index=sale_per_week.index, columns=columns
    )
    df_pred = df_pred.stack().to_frame("y_pred")
    df_pred.index.names = ["sku_id", "sale_date"]

    # 如果有取对数的话
    # df_test["y_true"] = np.expm1(df_test.y_true)
    # df_pred["y_pred"] = np.expm1(df_pred.y_pred)

    df_pred["y_pred"] = np.round(df_pred.y_pred, decimals=0)
    return df_test.join(df_pred)

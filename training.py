# -*- coding:utf-8 -*-
'''
author:zhangyu
date:2021/2/24
description:训练与预测
'''
from typing import Tuple, List

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import lightgbm as lgb
import matplotlib.pyplot as plt

from estimate import estimate
from feature_engineering import *
from load_data import *
from prediction_config import *


def training() -> Tuple[List[int], List[int]]:
    '''
        训练数据
    Returns:
        元组
    '''
    print("[INFO] Start training and predicting...")
    t0 = time.time()
    preds_train, preds_test = [], []
    ys_train, ys_test = [], []
    sale_per_week = get_sale_per_week()
    item_info = get_item_info(sale_per_week)
    for i in range(periods):
        print()
        print("# " + "=" * 100 + " #")
        print("# " + "Step %d" % (i + 1) + " " * (100 - len("Step %d" % (i + 1))) + " #")
        print("# " + "=" * 100 + " #")

        pred_dt = start_pred_dt + timedelta(weeks=i)
        training_dts = prepare_training_dts(pred_dt, gap=gap, rounds=3 * 7)
        X_train, y_train = prepare_training_set(
            sale_per_week, item_info, training_dts, gap=gap
        )
        X_test, y_test = prepare_val_set(sale_per_week, item_info, pred_dt, gap=gap)
        #     X_test = prepare_testing_set(val_per_hour, item_info, pred_dt, gap=gap)
        y_train[np.isnan(y_train)] = 0

        dtrain = lgb.Dataset(X_train, label=y_train)

        print("[INFO] Fit the model...")
        bst = lgb.train(
            params,
            dtrain,
            num_boost_round=MAX_ROUNDS,
            valid_sets=[dtrain],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=VERBOSE_EVAL,
        )

        # Predict
        pred_train = bst.predict(X_train, num_iteration=bst.best_iteration or MAX_ROUNDS)
        pred_test = bst.predict(X_test, num_iteration=bst.best_iteration or MAX_ROUNDS)

        # Store the intermediate results
        preds_train.append(pred_train)
        preds_test.append(pred_test)
        ys_train.append(y_train)
        ys_test.append(y_test)

        # Output feature importances
        feat_imps = sorted(
            zip(X_train.columns, bst.feature_importance("gain")),
            key=lambda x: x[1],
            reverse=True,
        )
        print()
        print("The feature importances are as follow: ")
        print(
            "\n".join(
                ["%s: %s" % (feat_name, feat_imp) for feat_name, feat_imp in feat_imps]
            )
        )

    print()
    print("[INFO] Finished! ( ^ _ ^ ) V")
    print("[INFO] Done in %f seconds." % (time.time() - t0))
    return ys_test, preds_test


def draw(result: pd.DataFrame) -> None:
    '''
       对pd进行画图
    Args:
        result: 预测结果
    Returns:
        None
    '''
    plt.figure(figsize=(6, 6))
    plt.title('prediction compare')
    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    plt.scatter(result["y_pred"], result["y_true"])
    plt.savefig("prediction.jpg")
    plt.show()


if __name__ == '__main__':
    # 预测结果
    ys_test, preds_test = training()
    # 评估算法
    result = estimate(ys_test, preds_test)
    result = result.reset_index(["sku_id", "sale_date"])
    # 误差计算
    print("[INFO] The MAE of testing set is:", mean_absolute_error(result.y_true, result.y_pred))
    # 均方根误差
    print("[INFO] The RMSE of testing set is:", mean_squared_error(result.y_true, result.y_pred) ** 0.5)
    # 决定系数
    print("[INFO] The R_quared score of testing set is:", r2_score(result.y_true, result.y_pred))
    # 可视化预测结果
    draw(result)

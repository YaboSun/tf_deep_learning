import os
import tarfile

import pandas as pd
import numpy as np

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "../datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    通过给定的数据路径将数据下载并保存到本地，首次运行执行
    """
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)

    tgz_path = os.path.join(housing_path, "housing.tgz")
    # 直接通过代码下载报错，connection refused，通过浏览器直接下载的压缩包
    # urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    """
    使用pandas加载数据
    """
    housing_csv_path = os.path.join(housing_path, "housing.csv")
    # 返回一个Pandas DataFrame对象
    return pd.read_csv(housing_csv_path)


def income_cat(data):
    """
        前面的方案都是随机抽样的方式，当数据集足够庞大的时候没有问题，但是如果不是的话会出现明显的抽# 样偏差，采用
        分层抽样解决，对收入中位数除以1.5来限制收入中位数的类别,使用ceil进行四舍五入，将大于5万的按5万处理
    """
    global strat_train_set, strat_test_set
    data["income_cat"] = np.ceil(data["median_income"] / 1.5)
    data["income_cat"].where(data["income_cat"] < 5, 5.0, inplace=True)

    # 使用sci-kit learn 的Stratified-Shuffle Split类进行分层抽样
    from sklearn.model_selection import StratifiedShuffleSplit
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data["income_cat"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

    return strat_train_set, strat_test_set
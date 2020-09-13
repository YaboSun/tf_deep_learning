import os
from six.moves import urllib
import tarfile
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import hashlib

"""
全局变量命名
"""
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    通过给定的数据路径将数据下载并保存到本地
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


def split_train_test(data, test_ratio):
    # 按照数据长度生成一个随机index序列，对应的数值为data中的行号
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    # iloc通过行号来获取数据
    return data.iloc[train_indices], data.iloc[test_indices]


# 上面解决方案存在一定问题，如果再次运行会生成不同的数据集，这样下去
# 机器学习算法将会看到整个数据集，而这是创建测试集过程中需要避免的
# 解决方案一
def test_set_check(identifier, test_ratio, hash):
    # 讲对应的hash值转化为64位二进制并取最后一位
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    # id_column 即创建的行号
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


# 解决方案二：方案一因为是使用添加新的行号为index列，所以在添加新数据时要确保在末尾添加，并且不会删除任何行
# 如果不能保证这点，可以使用属性中保证不变的属性作为列，一般来说一个地方的经纬度肯定不会发生变化，但是经纬度对应的精度没有那么精确，很可能好几个区域对应相同的经纬度


# 解决方案三：最简便使用scikit-learn中提供的原生函数进行切分，不过与前面的实现方式基本完全相同
def train_test_split(data):
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

    return train_set, test_set


# 前面的方案都是随机抽样的方式，当数据集足够庞大的时候没有问题，但是如果不是的话会出现明显的抽# 样偏差，采用分层抽样解决，对收入中位数除以1.5来限制收入中位数的类别
# 使用ceil进行四舍五入，将大于5万的按5万处理
def income_cat(data):
    data["income_cat"] = np.ceil(data["median_income"] / 1.5)
    data["income_cat"].where(data["income_cat"] < 5, 5.0, inplace=True)

    # 使用sci-kit learn 的Stratified-Shuffle Split类进行分层抽样
    from sklearn.model_selection import StratifiedShuffleSplit
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data["income_cat"]):
        strait_train_set = data.loc[train_index]
        strait_test_set = data.loc[test_index]
    
    return strait_train_set, strait_test_set


def drop_income_cat(strait_train_set, strait_test_set):
    for set in (strait_train_set, strait_test_set):
        set.drop(["income_cat"], axis=1, inplace=True)
    



if __name__ == "__main__":
    # fetch_housing_data()
    data = load_housing_data()
    # head()默认查看前5行，可传入指定行数
    # print(data.head(10))
    # 打印数据全部信息
    # print(data.info())
    # 打印最后一列所有类别的值统计
    # print(data["ocean_proximity"].value_counts())
    # 查看所有数值属性的摘要
    # print(data.describe())
    # 绘制每个数值属性的直方图Histogram，这个有点意思了
    # data.hist(bins=50, figsize=(20, 15))
    # plt.show()
    
    # 方案一：分割训练数据与测试数据
    # train_data, test_data = split_train_test(data, 0.2)
    
    # 方案二：利用hash生成id的方式分割训练数据与测试数据
    """
    需要可以唯一标识数据的id_column
    不幸的是这个数据没有这样的列，因此最简单的方式是通过每行行号进行创建
    """
    # housing_with_id = data.reset_index()
    # train_data, test_data = split_train_test_by_id(housing_with_id, 0.2, "index")
    # print(len(train_data))
    # print(len(test_data))

    # 方案三 采用原生sci api实现
    # train_data, test_data = train_test_split(data)
    # print(train_data.head())
    # print(test_data.head())

    # 房价中位数和收入中位数有很大关系，进行直方图分析
    # data["median_income"].hist()
    # plt.show()

    # 方案四：分层抽样
    # strait_train_set, strait_test_set = income_cat(data)
    # print(strait_train_set.head())
    # print(strait_test_set.head())
    # drop_income_cat(strait_train_set, strait_test_set)
    # print(data.head())
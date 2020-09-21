import hashlib

import matplotlib.pyplot as plt
import numpy as np

from housing_02 import data_utils

"""
全局变量命名
"""
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "../datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


class AnalyzeData:
    def __init__(self):
        self.housing_data = data_utils.load_housing_data()
        self.strat_train_set, self.strat_test_set = self.income_cat(self.housing_data)

    def split_train_test(self, data, test_ratio):
        """
        如果再次运行会生成不同的数据集，这样下去机器学习算法将会看到整个数据集，
        而这是创建测试集过程中需要避免的
        """
        # 按照数据长度生成一个随机index序列，对应的数值为data中的行号
        shuffled_indices = np.random.permutation(len(data))
        test_set_size = int(len(data) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        # iloc通过行号来获取数据
        return data.iloc[train_indices], data.iloc[test_indices]

    # 解决方案一
    def test_set_check(self, identifier, test_ratio, hash):
        # 将对应的hash值转化为64位二进制并取最后一位
        return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

    def split_train_test_by_id(self, data, test_ratio, id_column, hash=hashlib.md5):
        # id_column 即创建的行号
        ids = data[id_column]
        in_test_set = ids.apply(lambda id_: self.test_set_check(id_, test_ratio, hash))
        return data.loc[~in_test_set], data.loc[in_test_set]

    """
    解决方案二：方案一因为是使用添加新的行号为index列，所以在添加新数据时要确保在末尾添加，并且不会删除任何行
    如果不能保证这点，可以使用属性中保证不变的属性作为列，一般来说一个地方的经纬度肯定不会发生变化，但是经纬度
    对应的精度没有那么精确，很可能好几个区域对应相同的经纬度
    """

    # 解决方案三
    def train_test_split(self, data):
        """
        最简便使用scikit-learn中提供的原生函数进行切分，不过与前面的实现方式基本完全相同
        """
        from sklearn.model_selection import train_test_split
        train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

        return train_set, test_set

    def income_cat(self, data):
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

    def drop_income_cat(self, strait_train_set, strait_test_set):
        for set in (strait_train_set, strait_test_set):
            set.drop(["income_cat"], axis=1, inplace=True)

    def find_corr(self):
        """
        方法一：使用皮尔逊相关系数寻找相关性
        """
        corr_matrix = self.housing_data.corr()
        print(corr_matrix["median_house_value"].sort_values(ascending=False))
        """
        方法二：使用pandas的scatter_matrix函数绘制每个数值属性相对于其他属性的相关性
        """
        from pandas.plotting import scatter_matrix
        attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
        scatter_matrix(self.housing_data[attributes], figsize=(12, 8), diagonal="kde")
        plt.show()

        # 分析上图中最相关的可能是median_income，放大查看对应的散点图
        self.housing_data.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
        plt.show()

        """
        方法三：试验不同属性组合来分析相关性
        """
        self.housing_data["rooms_per_household"] = self.housing_data["total_rooms"] / self.housing_data["households"]
        self.housing_data["bedrooms_per_house"] = self.housing_data["total_bedrooms"] / self.housing_data["total_rooms"]
        self.housing_data["population_per_household"] = self.housing_data["population"] / self.housing_data["households"]
        corr_matrix = self.housing_data.corr()
        print(corr_matrix["median_house_value"].sort_values(ascending=False))


if __name__ == "__main__":
    prepare_data = AnalyzeData()
    data = prepare_data.housing_data
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
    # print(strait_train_set.head())
    # print(strait_test_set.head())

    # 数据可视化分析
    # housing_train = prepare_data.strat_train_set.copy()
    # print(len(housing_train))
    # 数据的地理分布图，设置alpha可以更清楚看到高密度数据点位置
    # housing_train.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    # plt.show()

    # 查看房价 scatter：分散图，利用名为jet的预定义颜色表进行可视化
    # housing_train.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    #                    s=housing_train["population"] / 100, label="population",
    #                    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
    # plt.legend()
    # plt.show()

    # 寻找相关性
    prepare_data.find_corr()

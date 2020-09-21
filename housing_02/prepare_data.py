from housing_02 import data_utils
from sklearn.impute import SimpleImputer


class PrepareData:
    def __init__(self):
        # 加载原始数据
        self.orig_housing = data_utils.load_housing_data()
        self.orig_strat_train_set, self.orig_strat_test_set = data_utils.income_cat(self.orig_housing)

        # 创建副本，将预测器和副本分开
        self.housing = self.orig_strat_train_set.drop("median_house_value", axis=1)
        self.housing_labels = self.orig_strat_train_set["median_house_value"].copy()

    def data_wash(self):
        """
        数据清洗，解决total_bedrooms属性部分缺失：
        1.放弃相应的地区
        2.放弃这个属性
        3.讲对应的缺失值设置为某个值（0， 平均数或者中位数）
        """
        # 方法一：放弃属性为空的地区
        self.housing.dropna(subset=["total_bed_rooms"])
        # 方法二：删掉对应的列，这里axis代表列，如果使用mean函数，使用的是求对应行的平均值
        self.housing.drop("total_bedrooms", axis=1)
        # 方法三：使用平均值填充空值，这个值需要保存下来，比如在重新评估系统时，需要更换测试集中的缺失值
        # 另外在系统上线时，需要使用新数据替换缺失值
        median = self.housing["total_bedrooms"].median()
        self.housing["total_bedrooms"].fillna(median)

    def data_wash_sklearn(self):
        """
        scikit-learn提供了一个简单的api来处理缺失值，SimpleImputer
        原书中提供的Imputer已经depricated
        """
        imputer = SimpleImputer(strategy="median")
        # 由于中位数值只能在数值属性上计算，所以创建一个没有ocean_proximity的副本
        housing_num = self.housing.drop("ocean_proximity", axis=1)
        # 使用fit方法将imputer适配到训练集
        imputer.fit(housing_num)
        print(imputer.statistics_)


if __name__ == '__main__':
    pre_data = PrepareData()
    pre_data.data_wash_sklearn()

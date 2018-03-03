# -*- coding: utf-8 -*-
"""
学习基本的 TensorFlow 概念
在 TensorFlow 中使用 LinearRegressor 类并基于单个输入特征预测各城市街区的房屋价值中位数
使用均方根误差 (RMSE) 评估模型预测的准确率
通过调整模型的超参数提高模型准确率
"""
import tensorflow as tf
import math
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from IPython import display
from sklearn import metrics
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

#加载数据集
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/ml_universities/california_housing_train.csv", sep=",")

#随机化处理数据
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe['median_house_value'] /= 1000.0

#检查数据,样本数、均值、标准偏差、最大值、最小值和各种分位数。
california_housing_dataframe.describe()
#
# #构建模型
#
# #定义特征，选取数值特征total_rooms
# my_feature = california_housing_dataframe[["total_rooms"]]
# #使用numeric_column定义特征列
# feature_columns = [tf.feature_column.numeric_column("total_rooms")]
#
# #定义目标标签
# targets = california_housing_dataframe["median_house_value"]
#
# #配置线性回归模型并使用 GradientDescentOptimizer（它会实现小批量随机梯度下降法 (SGD)）训练该模型。learning_rate 参数可控制梯度步长的大小。
# my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
# my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
# #为了安全起见，我们还会通过 clip_gradients_by_norm 将梯度裁剪应用到我们的优化器。梯度裁剪可确保梯度大小在训练期间不会变得过大，梯度过大会导致梯度下降法失败。
#
# #用特征列和控制器构成线性回归模型
# linear_regressor = tf.estimator.LinearRegressor(
#     feature_columns=feature_columns,
#     optimizer=my_optimizer
# )

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """
    定义输入函数，告诉tf如何进行数据预处理，以及训练模型是如何批处理，随机处理和重复数据。
    :param features: 特征，pandas DataFrame
    :param targets: 标签，目标pandas DataFrame
    :param batch_size: 指定 shuffle 将从中随机抽样的数据集的大小。
    :param shuffle: True or False，是不是在训练时随机输入传递到模型
    :param num_epochs: 周期数，重复数据，none表示数据无限重复
    :return: 构建迭代器，像LR返回下一批数据
    """
    # 将pandas数据转为array
    features = {key: np.array(value) for key, value in dict(features).items()}

    # 构建数据集, 将数据拆分成大小为 batch_size 的多批数据，以按照指定周期数 (num_epochs) 进行重复
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # 是否重复数据
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # 返回下一批数据
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
# #训练模型
# _ = linear_regressor.train(input_fn=lambda :my_input_fn(my_feature, targets),steps=100)
#
# #评估模型，在训练期间与数据的拟合情况
#
# #预测用的输入函数，对于每一个例子只做一次预测，不需要重复或者随机
# predictions_input_fn = lambda :my_input_fn(my_feature, targets, num_epochs=1,shuffle=False)
#
# #预测
# predictions = linear_regressor.predict(input_fn=predictions_input_fn)
# predictions = np.array([item['predictions'][0] for item in predictions])
# #均方误差以及均方根误差
# mean_squared_error = metrics.mean_squared_error(predictions, targets)
# root_mean_squared_error = math.sqrt(mean_squared_error)
# print("Mean Squared Error (on training data)： %0.3f"% mean_squared_error)
# print("Root Mean Squared Error( on training data ): %0.3f"%root_mean_squared_error)
#
# #比较RMSE与目标最大值最小值的差值
# min_house_value = california_housing_dataframe["median_house_value"].min()
# max_house_value = california_housing_dataframe["median_house_value"].max()
# min_max_difference = max_house_value - min_house_value
#
# print ("Min. Median House Value: %0.3f" % min_house_value)
# print ("Max. Median House Value: %0.3f" % max_house_value)
# print ("Difference between Min. and Max.: %0.3f" % min_max_difference)
# print ("Root Mean Squared Error: %0.3f"% root_mean_squared_error)
#
# #查看总体摘要统计信息
# calibration_data = pd.DataFrame()
# calibration_data["predictions"] = pd.Series(predictions)
# calibration_data["target"] = pd.Series(targets)
# describe = calibration_data.describe()
# print(describe)
#
# #可视化
# sample = california_housing_dataframe.sample(n=300)
# #total_rooms 的数据
# x_0 = sample["total_rooms"].min()
# x_1 = sample["total_rooms"].max()
# #训练时的偏差项和特征权重
# weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
# bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
#
# #对于rooms的最大最小值的y预测值
# y_0 = weight*x_0 + bias
# y_1 = weight*x_1 +bias
# plt.plot([x_0,x_1],[y_0,y_1],c='r')
# plt.ylabel("median_house_value")
# plt.xlabel("total_rooms")
# plt.scatter(sample["total_rooms"], sample["median_house_value"])
# plt.show()

#调整模型超参数
def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):
    """
    训练模型，寻找最好的模型超参数
    :param learning_rate: 学习率
    :param steps: 步长
    :param batch_size:每一次训练模型输入数量
    :param input_feature: 输入特征
    :return:
    """
    periods = 10
    steps_per_period = steps / periods

    my_feature = input_feature
    my_feature_data = california_housing_dataframe[[my_feature]]
    my_label = "median_house_value"
    targets = california_housing_dataframe[my_label]

    #构造特征列
    feature_columns = [tf.feature_column.numeric_column(my_feature)]

    #构造输入函数
    training_input_fn = lambda :my_input_fn(my_feature_data, targets,batch_size=batch_size)
    predictions_input_fn = lambda :my_input_fn(my_feature_data, targets,num_epochs=1,shuffle=False)

    #LR
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer=my_optimizer)
    #对于每个阶段的模型绘图
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title("Learn Line by Period")
    plt.ylabel(my_label)
    plt.xlabel(my_feature)
    sample =california_housing_dataframe.sample(n=300)
    plt.scatter(sample[my_feature],sample[my_label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1,1,periods)]

    print("Training model...")
    print("RMSE (on training data):")
    root_mean_squared_errors = []
    for period in range(0, periods):
        # 训练每个阶段的模型
        linear_regressor.train(input_fn=training_input_fn, steps=steps_per_period)
        # 预测
        predictions = linear_regressor.predict(input_fn=predictions_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])

        # 计算损失
        root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(predictions, targets))
        # 输出当前损失
        print("  period %02d : %0.2f" % (period, root_mean_squared_error))
        # 将当前阶段的损失加入list中
        root_mean_squared_errors.append(root_mean_squared_error)
        # Finally, track the weights and biases over time.
        # Apply some math to ensure that the data and line are plotted neatly.
        y_extents = np.array([0, sample[my_label].max()])

        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.minimum(x_extents,
                                          sample[my_feature].max()),
                               sample[my_feature].min())
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color=colors[period])
    print("Model training finished.")

    # 输出每个阶段的损失矩阵
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)
    plt.show()

    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    display.display(calibration_data.describe())

    print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)

train_model(learning_rate=0.00002, steps=500, batch_size=5)



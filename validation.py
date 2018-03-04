# -*- coding: utf-8 -*-
"""
使用多个特征而非单个特征来进一步提高模型的有效性
调试模型输入数据中的问题
使用测试数据集检查模型是否过拟合验证数据
"""
import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from tensorflow_first_learn import LinearRe

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
#加载数据
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/ml_universities/california_housing_train.csv", sep=",")
#数据随机化
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index)
)
def preprocess_feature(dataframe):
    """
    处理输入数据为我们需要的特征
    :param dataframe: 输入的额数据集，在这里是califonia_housing,为pandas dataframe
    :return: dataframe，包含模型需要的特征
    """
    selected_features = dataframe[
        [
            "latitude",
            "longitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income"
        ]
    ]
    prcessed_feature = selected_features.copy()
    #合成特征
    prcessed_feature["rooms_per_person"] = (
        dataframe["total_rooms"]/dataframe["population"]
    )
    return prcessed_feature

def preprocess_targets(dataframe):
    """
    处理数据为我们需要的目标标签
    :param dataframe: 输入的额数据集，在这里是califonia_housing,为pandas dataframe
    :return: dataframe，目标标签
    """
    output_targets = pd.DataFrame()
    #正规化处理itargets，把数值统一为k级
    output_targets["median_house_value"] = (
        dataframe["median_house_value"]/1000.0
    )
    return output_targets
#构建训练集
training_examples = preprocess_feature(california_housing_dataframe.head(12000))
# print(training_examples.describe())

training_targets = preprocess_targets(california_housing_dataframe.head(12000))
# print(training_targets.describe())

#构建验证集
validation_examples = preprocess_feature(california_housing_dataframe.tail(5000))
# print(validation_examples.describe())

validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
# print(validation_targets.describe())

# #绘制纬度/经度与房屋价值中位数的曲线图
# plt.figure(figsize=(13, 8))
#
# ax = plt.subplot(1, 2, 1)
# ax.set_title("Validation Data")
#
# ax.set_autoscaley_on(False)
# ax.set_ylim([32, 43])
# ax.set_autoscalex_on(False)
# ax.set_xlim([-126, -112])
# plt.scatter(validation_examples["longitude"],
#             validation_examples["latitude"],
#             cmap="coolwarm",
#             c=validation_targets["median_house_value"]/validation_targets["median_house_value"].max())
# ax = plt.subplot(1,2,2)
# ax.set_title("Training Data")
#
# ax.set_autoscaley_on(False)
# ax.set_ylim([32, 43])
# ax.set_autoscalex_on(False)
# ax.set_xlim([-126, -112])
# plt.scatter(training_examples["longitude"],
#             training_examples["latitude"],
#             cmap="coolwarm",
#             c=training_targets["median_house_value"] / training_targets["median_house_value"].max())
# plt.show()
lr = LinearRe()
LINEAR_REGRESSOR = lr.train_model_moreFeatures(
    learning_rate=0.00003,
    steps=500,
    batch_size=5,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets
)
#测试
california_housing_test_data = pd.read_csv("https://storage.googleapis.com/ml_universities/california_housing_test.csv", sep=",")

test_examples = preprocess_feature(california_housing_test_data)
test_targets = preprocess_targets(california_housing_test_data)

predict_test_input_fn = lambda: lr.my_input_fn(
      test_examples,
      test_targets["median_house_value"],
      num_epochs=1,
      shuffle=False)

test_predictions = LINEAR_REGRESSOR.predict(input_fn=predict_test_input_fn)
test_predictions = np.array([item['predictions'][0] for item in test_predictions])

root_mean_squared_error = math.sqrt(
    metrics.mean_squared_error(test_predictions, test_targets))

print ("Final RMSE (on test data): %0.2f" % root_mean_squared_error)

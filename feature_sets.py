# -*- coding:utf-8 -*-
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

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/ml_universities/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
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
print(training_examples.describe())

training_targets = preprocess_targets(california_housing_dataframe.head(12000))
print(training_targets.describe())

#构建验证集
validation_examples = preprocess_feature(california_housing_dataframe.tail(5000))
print(validation_examples.describe())

validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
print(validation_targets.describe())
"""
构建良好的特征集
相关矩阵展现了两两比较的相关性，既包括每个特征与目标特征之间的比较，也包括每个特征与其他特征之间的比较。
在这里，相关性被定义为皮尔逊相关系数。
相关性值具有以下含义：
-1.0：完全负相关
0.0：不相关
1.0：完全正相关
"""
correlation_dataframe = training_examples.copy()
correlation_dataframe["target"] = training_targets["median_house_value"]

correlation_dataframe.corr()


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
def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])

def train_model_moreFeatures( learning_rate, steps, batch_size,
                             training_examples, training_targets, validation_examples,validation_targets):
    """

    :param learning_rate:
    :param step:
    :param batch_size:
    :param training_examples:
    :param training_targets:
    :param validation_examples:
    :param validation_targets:
    :return:
    """
    periods = 10
    steps_per_period = steps / periods

    # Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=my_optimizer
    )

    # Create input functions
    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets["median_house_value"],
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                    training_targets["median_house_value"],
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                      validation_targets["median_house_value"],
                                                      num_epochs=1,
                                                      shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period,
        )
        # Take a break and compute predictions.
        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.show()

    return linear_regressor
"""
理想情况下，我们希望具有与目标密切相关的特征。
此外，我们还希望有一些相互之间的相关性不太密切的特征，以便它们添加独立信息。
利用这些信息来尝试移除特征。您也可以尝试构建其他合成特征，例如两个原始特征的比例。
"""
minimal_features = [
    "median_income",
    # "rooms_per_person",
    # "housing_median_age"
    "latitude"
]

assert minimal_features, "You must select at least one feature!"

minimal_training_examples = training_examples[minimal_features]
minimal_validation_examples = validation_examples[minimal_features]

#
# Don't forget to adjust these parameters.
#
# train_model_moreFeatures(
#     learning_rate=0.01,
#     steps=500,
#     batch_size=5,
#     training_examples=minimal_training_examples,
#     training_targets=training_targets,
#     validation_examples=minimal_validation_examples,
#     validation_targets=validation_targets)
#更好地利用纬度
#画图查看两个特征的关系，不存在线性
plt.figure(2)
plt.scatter(training_examples["latitude"], training_targets["median_house_value"])
plt.show()
"""
可以创建某个特征，将 latitude 映射到值 |latitude - 38|，并将该特征命名为 distance_from_san_francisco。
或者，您可以将该空间分成 10 个不同的分桶（例如 latitude_32_to_33、latitude_33_to_34 等）：如果 latitude 位于相应分桶范围内，则显示值 1.0；如果不在范围内，则显示值 0.0。
使用相关矩阵来指导构建合成特征；如果发现效果还不错的合成特征，可以将其添加到模型中
"""


def select_and_transform_features(source_df):
  LATITUDE_RANGES = zip(range(32, 44), range(33, 45))
  selected_examples = pd.DataFrame()
  selected_examples["median_income"] = source_df["median_income"]
  for r in LATITUDE_RANGES:
    selected_examples["latitude_%d_to_%d" % r] = source_df["latitude"].apply(
      lambda l: 1.0 if l >= r[0] and l < r[1] else 0.0)
  return selected_examples

selected_training_examples = select_and_transform_features(training_examples)
print(selected_training_examples.head())
selected_validation_examples = select_and_transform_features(validation_examples)

train_model_moreFeatures(
    learning_rate=0.01,
    steps=500,
    batch_size=5,
    training_examples=selected_training_examples,
    training_targets=training_targets,
    validation_examples=selected_validation_examples,
    validation_targets=validation_targets)

# -*- coding:utf-8 -*-

"""
逻辑回归
学习目标：
将（在之前的练习中构建的）房屋价值中位数预测模型重新构建为二元分类模型
比较逻辑回归与线性回归解决二元分类问题的有效性
与在之前的练习中一样，我们将使用加利福尼亚州住房数据集，但这次我们会预测某个城市街区的住房成本是否高昂，从而将其转换成一个二元分类问题。此外，我们还会暂时恢复使用默认特征。

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

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))

def preprocess_features(california_housing_dataframe):

    selected_features = california_housing_dataframe[
        ["latitude",
         "longitude",
         "housing_median_age",
         "total_rooms",
         "total_bedrooms",
         "population",
         "households",
         "median_income"]]
    processed_features = selected_features.copy()
    # Create a synthetic feature.
    processed_features["rooms_per_person"] = (
        california_housing_dataframe["total_rooms"] /
        california_housing_dataframe["population"])
    return processed_features

def preprocess_targets(california_housing_dataframe):

  output_targets = pd.DataFrame()

  output_targets["median_house_value_is_high"] = (
    california_housing_dataframe["median_house_value"] > 265000).astype(float)
  return output_targets

training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))


validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))


print("Training examples summary:")
display.display(training_examples.describe())
print ("Validation examples summary:")
display.display(validation_examples.describe())

print ("Training targets summary:")
display.display(training_targets.describe())
print ("Validation targets summary:")
display.display(validation_targets.describe())

def construct_feature_columns(input_features):

  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):


    features = {key: np.array(value) for key, value in dict(features).items()}


    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)


    if shuffle:
        ds = ds.shuffle(10000)


    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_linear_regressor_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):

    periods = 10
    steps_per_period = steps / periods


    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=my_optimizer
    )


    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets["median_house_value_is_high"],
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                    training_targets["median_house_value_is_high"],
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                      validation_targets["median_house_value_is_high"],
                                                      num_epochs=1,
                                                      shuffle=False)


    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):

        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )


        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])


        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))

        print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))

        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print("Model training finished.")


    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.show()

    return linear_regressor

# linear_regressor = train_linear_regressor_model(
#     learning_rate=0.000001,
#     steps=200,
#     batch_size=20,
#     training_examples=training_examples,
#     training_targets=training_targets,
#     validation_examples=validation_examples,
#     validation_targets=validation_targets)
#
# predict_validation_input_fn = lambda: my_input_fn(validation_examples,
#                                                   validation_targets["median_house_value_is_high"],
#                                                   num_epochs=1,
#                                                   shuffle=False)
#
# validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
# validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
#
# plt.figure(2)
# plt.hist(validation_predictions)
# plt.show()


def train_linear_classifier_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """
    逻辑回归模型

    Args:
      learning_rate: A `float`, the learning rate.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      training_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for training.
      training_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for training.
      validation_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for validation.
      validation_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for validation.

    Returns:
      A `LinearClassifier` object trained on the training data.
    """

    periods = 10
    steps_per_period = steps / periods

    # Create a linear classifier object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_classifier =  tf.estimator.LinearClassifier(feature_columns=construct_feature_columns(training_examples)
                                                       , optimizer=my_optimizer)

    # Create input functions
    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets["median_house_value_is_high"],
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                    training_targets["median_house_value_is_high"],
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                      validation_targets["median_house_value_is_high"],
                                                      num_epochs=1,
                                                      shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("LogLoss (on training data):")
    training_log_losses = []
    validation_log_losses = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
        training_probabilities = np.array([item['probabilities'] for item in training_probabilities])

        validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
        validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

        training_log_loss = metrics.log_loss(training_targets, training_probabilities)
        validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_log_loss))
        # Add the loss metrics from this period to our list.
        training_log_losses.append(training_log_loss)
        validation_log_losses.append(validation_log_loss)
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.tight_layout()
    plt.plot(training_log_losses, label="training")
    plt.plot(validation_log_losses, label="validation")
    plt.legend()
    plt.show()

    return linear_classifier


linear_classifier = train_linear_classifier_model(
    learning_rate=0.000005,
    steps=500,
    batch_size=20,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
"""
分类时非常有用的一些指标包括：模型准确率、ROC 曲线和 ROC 曲线下面积 (AUC)。我们会检查这些指标。
LinearClassifier.evaluate 可计算准确率和 AUC 等实用指标。
"""
predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                      validation_targets["median_house_value_is_high"],
                                                      num_epochs=1,
                                                      shuffle=False)
evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)

print ("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
print ("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])

"""
您可以使用类别概率（例如由 LinearClassifier.predict 和 Sklearn 的 roc_curve 计算的概率）
来获得绘制 ROC 曲线所需的真正例率和假正例率。
"""

# Get just the probabilities for the positive class
validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
validation_probabilities = np.array([item['probabilities'][1] for item in validation_probabilities])

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
    validation_targets, validation_probabilities)
plt.plot(false_positive_rate, true_positive_rate, label="our model")
plt.plot([0, 1], [0, 1], label="random classifier")
_ = plt.legend(loc=2)

linear_classifier = train_linear_classifier_model(
    learning_rate=0.000003,
    steps=20000,
    batch_size=500,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)

print ("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
print ("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])

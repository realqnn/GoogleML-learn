# -*- coding: utf-8 -*-
"""
合成特征和离群值
创建一个合成特征，即另外两个特征的比例
将此新特征用作线性回归模型的输入
通过识别和截取（移除）输入数据中的离群值来提高模型的有效性
"""
from tensorflow_first_learn import *


if __name__ == '__main__':
    Linear_model = LinearRe()
    """
    任务1：尝试合成新特征
    total_rooms 和 population 特征都会统计指定街区的相关总计数据。
    但是，如果一个街区比另一个街区的人口更密集，会怎么样？我们可以创建一个合成特征（即 total_rooms 与 population 的比例）来探索街区人口密度与房屋价值中位数之间的关系。
    创建一个名为 rooms_per_person 的特征，并将其用作 train_model() 的 input_feature。
    通过调整学习速率，您使用这一特征可以获得的最佳效果是什么？（效果越好，回归线与数据的拟合度就越高，最终 RMSE 也会越低。）
    """
    california_housing_dataframe["room_per_perperson"] = \
        california_housing_dataframe["total_rooms"]/california_housing_dataframe["population"]
    california_data = Linear_model.train_model(
        learning_rate=0.005,
        steps=500,
        batch_size=5,
        input_feature="room_per_perperson"
    )
    """
    识别离群值
    我们可以通过创建预测值与目标值的散点图来可视化模型效果。理想情况下，这些值将位于一条完全相关的对角线上。
   使用您在任务 1 中训练过的人均房间数模型，并使用 Pyplot 的 scatter() 创建预测值与目标值的散点图。
   您是否看到任何异常情况？通过查看 rooms_per_person 中值的分布情况，将这些异常情况追溯到源数据
    
    """
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(california_data["predictions"],california_data["targets"])
    plt.show()
    plt.subplot(1, 2, 2)
    _ = california_housing_dataframe["room_per_perperson"].hist()
    #直方图显示，大多数值都小于 5。我们将 rooms_per_person 的值截取为 5，然后绘制直方图以再次检查结果
    california_housing_dataframe["rooms_per_person"] = (
        california_housing_dataframe["rooms_per_person"]).apply(lambda x: min(x, 5))

    _ = california_housing_dataframe["rooms_per_person"].hist()
    #再次验证结果
    calibration_data = Linear_model.train_model(
        learning_rate=0.05,
        steps=500,
        batch_size=5,
        input_feature="rooms_per_person")
    _ = plt.scatter(calibration_data["predictions"], calibration_data["targets"])
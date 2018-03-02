# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib as mlb
import numpy as np
print pd.__version__

"""
pandas 中的主要数据结构被实现为以下两类：
DataFrame，您可以将它想象成一个关系型数据表格，其中包含多个行和已命名的列。
Series，它是单一列。DataFrame 中包含一个或多个 Series，每个 Series 均有一个名称。
"""
#创建Series
s1 = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
print s1

#可以将映射 string 列名称的 dict 传递到它们各自的 Series，从而创建DataFrame对象。如果 Series 在长度上不一致，系统会用特殊的 NA/NaN 值填充缺失的值。
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
data = pd.DataFrame({'City name': city_names, 'Population':population })
print city_names
print population
print data

"""
需要将整个文件加载到 DataFrame 中。下面的示例加载了一个包含加利福尼亚州住房数据的文件。
请运行以加载数据，并创建特征定义：
"""
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/ml_universities/california_housing_train.csv", sep=",")

california_housing_dataframe.describe()#显示DataFrame的统计信息
print california_housing_dataframe.describe()
print california_housing_dataframe.head() #显示前几个记录
#pandas 的另一个强大功能是绘制图表。例如，借助 DataFrame.hist，可以快速了解一个列中值的分布

california_housing_dataframe.hist('housing_median_age')
#使用python dict/list 指定访问datafram数据
cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
print type(cities['City name'])
print cities['City name']
print type(cities['City name'][1])
print cities['City name'][1]
print type(cities[0:2])
print cities[0:2]

#操控数据
print population / 1000.
print np.log(population)

"""
对于更复杂的单列转换，您可以使用 Series.apply。像 Python 映射函数一样，Series.apply 将以参数形式接受 lambda 函数，而该函数会应用于每个值。
下面的示例创建了一个指明 population 是否超过 100 万的新 Series：
"""
newpopulation = population.apply(lambda val: val > 1000000)
print newpopulation
#修改dataframe
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
print cities
cities['if name used person'] = cities['City name'].apply(lambda name: name.startswith('San'))
cities['city square miles >50'] = cities['Area square miles'].apply(lambda val: val >= 50)
print cities
"""
Series 和 DataFrame 对象也定义了 index 属性，该属性会向每个 Series 项或 DataFrame 行赋一个标识符值。
默认情况下，在构造时，pandas 会赋可反映源数据顺序的索引值。索引值在创建后是稳定的；也就是说，它们不会因为数据重新排序而发生改变。
"""
print city_names.index
print cities.index
#调用 DataFrame.reindex 以手动重新排列各行的顺序。例如，以下方式与按城市名称排序具有相同的效果：
print cities.reindex([2, 0, 1])
#随机重建索引
cities.reindex(np.random.permutation(cities.index))
"""
 reindex 输入数组包含原始 DataFrame 索引值中没有的值，reindex 会为此类“丢失的”索引添加新行，并在所有对应列中填充 NaN 值
"""
print cities.reindex([0, 4, 5, 2])
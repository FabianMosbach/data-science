import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.tree import export_graphviz
import seaborn as sb
import numpy as np
import graphviz

#===============================================================================
# Splittet einen komplette Datensatz in 4 Datensaetze fuer Algorithmen
#===============================================================================
def split(p_dataset, p_split_location, p_test_size):   
    samples, features = np.split(p_dataset, [p_split_location], axis=1)   
    samples_train, samples_test, features_train, features_test = model_selection.train_test_split(samples, features, test_size = p_test_size)
    return samples_train, samples_test, features_train, features_test

#===============================================================================
# MAIN
#===============================================================================
url = 'https://raw.githubusercontent.com/rhiever/Data-Analysis-and-Machine-Learning-Projects/master/example-data-science-notebook/iris-data.csv'

iris_data = pd.read_csv(url, sep=',', na_values=['NA'])

print(iris_data.head())
print(iris_data.describe())

iris_data.loc[iris_data['class'] == 'versicolor', 'class'] = 'Iris-versicolor'
iris_data.loc[iris_data['class'] == 'Iris-setossa', 'class'] = 'Iris-setosa'

iris_data['class'].unique()

iris_data = iris_data.loc[(iris_data['class'] != 'Iris-setosa') | (iris_data['sepal_width_cm'] >= 2.5)]
iris_data.loc[iris_data['class'] == 'Iris-setosa', 'sepal_width_cm'].hist()

iris_data.loc[(iris_data['class'] == 'Iris-versicolor') &
              (iris_data['sepal_length_cm'] < 1.0),
              'sepal_length_cm'] *= 100.0

iris_data.loc[iris_data['class'] == 'Iris-versicolor', 'sepal_length_cm'].hist()

print(iris_data.loc[(iris_data['sepal_length_cm'].isnull()) |
              (iris_data['sepal_width_cm'].isnull()) |
              (iris_data['petal_length_cm'].isnull()) |
              (iris_data['petal_width_cm'].isnull())])

iris_data.loc[iris_data['class'] == 'Iris-setosa', 'petal_width_cm'].hist()



#===============================================================================
# Imputing Data
#===============================================================================
average_petal_width = iris_data.loc[iris_data['class'] == 'Iris-setosa', 'petal_width_cm'].mean()

iris_data.loc[(iris_data['class'] == 'Iris-setosa') &
              (iris_data['petal_width_cm'].isnull()),
              'petal_width_cm'] = average_petal_width

iris_data.loc[(iris_data['class'] == 'Iris-setosa') &
              (iris_data['petal_width_cm'] == average_petal_width)]

#===============================================================================
# Alternatively dropping data
#===============================================================================
#------------------------------------------------ iris_data.dropna(inplace=True)


#===============================================================================
# Write clean Data File
#===============================================================================
iris_data.to_csv('iris-data-clean.csv', index=False)
iris_data_clean = pd.read_csv('iris-data-clean.csv')

#===============================================================================
# Decision Tree Classification
#===============================================================================
dtc = DecisionTreeClassifier()

samples_train, samples_test, features_train, features_test = split(iris_data_clean, 4, 0.5)

dtc.fit(samples_train, features_train)

print(dtc.feature_importances_)

predict_test = dtc.predict(samples_test)
idpred = pd.DataFrame(data=predict_test, index=samples_test.index.values) 

test_set = pd.concat([samples_test, features_test], axis=1)
prediction_set = pd.concat([test_set , idpred], axis=1)

export_graphviz(dtc, out_file='tree.dot',  feature_names=iris_data_clean.columns.values,  
                         class_names=iris_data_clean['class'].unique(),
                         label='none',
                         filled=True, rounded=True,  
                         special_characters=True)
#graphviz.Source(export_graphviz(dtc, None))
graphviz.render('dot', 'png', 'tree.dot')


print(classification_report(features_test, predict_test))

#===============================================================================
# Happy plotting
#===============================================================================
plt.figure(figsize=(10, 10))

for column_index, column in enumerate(iris_data_clean.columns):
    if column == 'class':
        continue
    plt.subplot(2, 2, column_index + 1)
    sb.violinplot(x='class', y=column, data=iris_data_clean)
    
sb.pairplot(iris_data.dropna(), hue='class')
plt.show()


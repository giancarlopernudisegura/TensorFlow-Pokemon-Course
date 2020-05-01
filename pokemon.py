import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


df = pd.read_csv('./pokemon_alopez247.csv')
df = df[['isLegendary','Generation', 'Type_1', 'Type_2', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed','Color','Egg_Group_1','Height_m','Weight_kg','Body_Style']]


df['isLegendary'] = df['isLegendary'].astype(int)

def dummy_creation(df, dummy_categories):
    for i in dummy_categories:
        df_dummy = pd.get_dummies(df[i])
        df = pd.concat([df,df_dummy],axis=1)
        df = df.drop(i, axis=1)
    return(df)

df = dummy_creation(df, ['Egg_Group_1', 'Body_Style', 'Color', 'Type_1', 'Type_2'])


def split_data(DataFrame, column):
    df_train = DataFrame.loc[df[column] != 1]
    df_test = DataFrame.loc[df[column] == 1]

    df_train = df_train.drop(column, axis=1)
    df_test = df_test.drop(column, axis=1)

    return (df_train, df_test)

df_train, df_test = split_data(df, 'Generation')

def label_delineator(df_train, df_test, label):
    train_data = df_train.drop(label, axis=1).values
    train_labels = df_train[label].values
    test_data = df_test.drop(label, axis=1).values
    test_labels = df_test[label].values

    return (train_data, train_labels, test_data, test_labels)

train_data, train_labels, test_data, test_labels = label_delineator(df_train, df_test, 'isLegendary')


train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
test_data = preprocessing.MinMaxScaler().fit_transform(test_data)

length = train_data.shape[1]

model = keras.Sequential()
model.add(keras.layers.Dense(500, activation='relu', input_shape=[length,]))
model.add(keras.layers.Dense(2, activation='softmax'))

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=500)

loss_value, accuracy_value = model.evaluate(test_data, test_labels)
print(f'Test Accuracy: {accuracy_value}')

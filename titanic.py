import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import sys
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')


path = "/Users/teddyl/tensorflow/dataset/"

test_df = pd.read_csv(path + "test.csv",index_col=0)
train_df = pd.read_csv(path + "train.csv",index_col=0)
# Survived : 생존 여부(종속 변수)
# 0 = 사망
# 1 = 생존
# Pclass : 1,high,2,3low
# Name : 이름
# Sex : 성별
# Age : 나이
# SibSp : 동반한 Sibling(형제자매)와 Spouse(배우자)의 수
# Parch : 동반한 Parent(부모) Child(자식)의 수
# Ticket : 티켓의 고유넘버
# Fare : 티켓의 요금
# Cabin : 객실 번호
# Embarked : 승선한 항
# combine=[test_df,train_df]
print(train_df.columns.values)
print(train_df.head()) #11 columns
print(train_df.info())
print(test_df.info())
print(train_df.describe())
print(train_df.describe(include=['O']))

print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# g = sns.FacetGrid(train_df, col='Survived')
# g.map(plt.hist, 'Age', bins=20)
# plt.show()
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', height=2.2, aspect=1.6)

# 바그래프로 시각화, x: 성별, y: 요금, Error bar: 표시 안 함
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None,order=["male","female"])

grid.add_legend()
# plt.show()
print("Before", train_df.shape, test_df.shape)# (891, 11) (418, 10)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1) # ticket(티켓 고유 넘버), cabin(객실 고유넘버) 불필요로 제거 
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1) # ticket(티켓 고유 넘버), cabin(객실 고유넘버) 불필요로 제거
combine = [train_df, test_df]
print("After", train_df.shape, test_df.shape)#(891, 9) (418, 8)

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

print(train_df.head())

# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:22:52 2026

@author: angel
""" 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

"""
Declaro la función downsample donde se podran realizar una 
disminución de muestras.
Se introduciran las caracteristicas, los objetivo y la cantidad 
de veces que incrementara, para obtener datos más equilibrados 
y revueltos con la función shuffle.
"""
def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_downsampled = pd.concat(
        [features_zeros.sample(frac=fraction, random_state=12345)]
        + [features_ones]
    )
    target_downsampled = pd.concat(
        [target_zeros.sample(frac=fraction, random_state=12345)]
        + [target_ones]
    )

    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=12345
    )

    return features_downsampled, target_downsampled

"""
Declaro la función upsample donde se podran realizar un aumento 
de muestras.
Se introduciran las caracteristicas, los objetivo y la cantidad 
de veces que incrementara, para obtener datos más equilibrados 
y revueltos con la función shuffle.
"""
def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345
    )

    return features_upsampled, target_upsampled

df1 = pd.read_csv('datasets/application_record.csv')
df2 = pd.read_csv('datasets/credit_record.csv')

#print(df1.info())
#print(df2.info())

df1.columns = df1.columns.str.lower()
df2.columns = df2.columns.str.lower()

mask = df1['days_employed'] > 0
df1.loc[mask,'occupation_type'] = df1.loc[mask,'occupation_type'].fillna('unployed')
df1.fillna('unknown',inplace=True)

status_map = {
    "X": 0,
    "C": 0,
    "0": 1,
    "1": 2,
    "2": 3,
    "3": 4,
    "4": 5,
    "5": 6
}

df2["status_score"] = df2["status"].map(status_map)

df2['target'] = np.where(df2['status_score'] >= 2, 1, 0)

merged = pd.merge(df1, df2[["id",'status_score', "target"]], on="id", how="inner")

features = merged.drop(['id','target'],axis=1)
target = merged['target']

features_train_1, features_retest, target_train_1, target_retest = train_test_split(
    features,
    target,
    test_size=0.1,
    random_state=12345
)

features_train, features_test, target_train, target_test = train_test_split(
    features_train_1,
    target_train_1,
    test_size=0.2,
    random_state=12345
)

features_upsample, target_upsample = upsample(
    features_train, target_train, 2 
)

features_downsample, target_downsample = downsample(
    features_upsample, target_upsample, 0.12)

print(target_downsample.value_counts())


fig, axs = plt.subplots(len(df1.columns),1, figsize=(10,120))
for i in range(len(df1.columns)):
    axs[i].hist(df1[df1.columns[i]])
    axs[i].set_title(df1.columns[i])

plt.figure(2,figsize=(40,4))
plt.hist(df1['occupation_type'],bins=20)
plt.title('Tipos de ocupaciones')


fig3, axs3 = plt.subplots(2,1,figsize=(6,4))
axs3[0].hist(df2['months_balance'])
axs3[1].hist(df2['status_score'])

fig4, axs4 = plt.subplots(2,1,figsize=(4,6))
axs4[0].boxplot(df2['months_balance'])
axs4[1].boxplot(df2['status_score'])

plt.figure(5)
plt.hist(target_downsample)
plt.title('Equilibrio de datos')
plt.show()

   
plt.show()
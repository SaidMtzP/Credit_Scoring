# -*- coding: utf-8 -*-
"""
Created on Wend May  4 10:54:20 2026

@author: angel
"""

import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score

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

#print(f'first Dataset:\nNan count:\n{df1.isnull().sum()}, Duplicates count:{df1.duplicated().sum()}\nSecond Dataset\nNan count:\n{df2.isnull().sum()}, Duplicates count:{df2.duplicated().sum()}')


#fig, axs = plt.subplots(len(df1.columns),1, figsize=(10,120))
#for i in range(len(df1.columns)):
#    axs[i].hist(df1[df1.columns[i]])
#    axs[i].set_title(df1.columns[i])

#plt.figure(figsize=(40,4))
#plt.hist(df1['occupation_type'],bins=20)
#plt.title('Tipos de ocupaciones')

#fig2, axs2 = plt.subplots(2,1,figsize=(6,4))
#axs2[0].hist(df2['months_balance'])
#axs2[1].hist(df2['status'])

    
#plt.show()


# i.	Lowest Risk: C (Paid off that month) and X (No loan for the month)
# ii.	Status 0: 1-29 days past due
# iii.	Status 1: 30-59 days past due
# iv.	Status 2: 60-89 days past due
# v.	Status 3: 90-119 days past due
# vi.	Status 4: 120-149 days past due
# vii.	Highest Risk: Status 5 (Overdue or bad debts, write-offs for more than 150 days)

# Define severity mapping
status_map = {
    "X": -1,
    "C": 0,
    "0": 1,
    "1": 2,
    "2": 3,
    "3": 4,
    "4": 5,
    "5": 6
}

df2["status_score"] = df2["status"].map(status_map)

#print(df2.head())

#fig2, axs2 = plt.subplots(2,1,figsize=(6,4))
#axs2[0].hist(df2['months_balance'])
#axs2[1].hist(df2['status_score'])

#fig3, axs3 = plt.subplots(2,1,figsize=(4,6))
#axs3[0].boxplot(df2['months_balance'])
#axs3[1].boxplot(df2['status_score'])
    
#plt.show()

df2['target'] = np.where(df2['status_score'] >= 2, 1, 0)

#merged = pd.merge(df1, df2[["id",'status_score', "target"]], on="id", how="inner")
merged = pd.merge(df1, df2[["id", "target"]], on="id", how="inner")

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


features_upsample_train, target_upsample_train = upsample(
    features_train, target_train, 2 )
features_downsample_train, target_downsample_train = downsample(
    features_upsample_train, target_upsample_train, 0.12)



features_upsample_test, target_upsample_test = upsample(
    features_test, target_test, 2 )
features_downsample_test, target_downsample_test = downsample(
    features_upsample_test, target_upsample_test, 0.12)



features_upsample_retest, target_upsample_retest = upsample(
    features_retest, target_retest, 2 )
features_downsample_retest, target_downsample_retest = downsample(
    features_upsample_retest, target_upsample_retest, 0.12)


params = {
    'objective': 'binary',
    'metric': 'auc',
    #'is_unbalance': True,          # <--- Clave para el desbalanceo automático
    'scale_pos_weight': 4,
    'learning_rate': 0.03,
    'num_leaves': 51,
    'verbose': -1
}

columnas_texto = [
    'code_gender', 
    'flag_own_car',
    'flag_own_realty',
    'name_income_type',
    'name_education_type',
    'name_family_status',
    'name_housing_type',
    'occupation_type'
    ]

for col in columnas_texto:
    features_downsample_train[col] = features_downsample_train[col].astype('category')
    features_downsample_test[col] = features_downsample_test[col].astype('category')
    features_downsample_retest[col] = features_downsample_retest[col].astype('category')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(features_downsample_train, target_downsample_train)):
    X_train_fold, X_val_fold = features_downsample_train.iloc[train_idx], features_downsample_train.iloc[val_idx]
    y_train_fold, y_val_fold = target_downsample_train.iloc[train_idx], target_downsample_train.iloc[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    test_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
    #retest_data = lgb.Dataset.create_valid(features_downsample_retest, label=target_downsample_retest, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=100)]
    )

    pred_probs = model.predict(X_val_fold)
    auc_scores.append(roc_auc_score(y_val_fold, pred_probs))
    
print(f"ROC-AUC Score: {np.mean(auc_scores):.4f}")
#final_score = (cross_val_score(model, train_data).sum())/5
#print('Puntuación media de la evaluación del modelo:', final_score)
#reporte = classification_report(target_downsample_retest, pred_probs)
#print(reporte)

#print(train_data)





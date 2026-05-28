# -*- coding: utf-8 -*-
"""
Created on Wend May  4 10:54:20 2026

@author: angel
"""
# Importacion de librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Importacion de librerias de sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, auc

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

# Se cargan  los datasets
df1 = pd.read_csv('datasets/application_record.csv')  # Dataset info. de usuarios
df2 = pd.read_csv('datasets/credit_record.csv')  # Dartaset info. record crediticio

#print(df1.info())
#print(df2.info())

# Se cambian las columnas a puras letras minusculas por facilidad
df1.columns = df1.columns.str.lower()
df2.columns = df2.columns.str.lower()

# Se va a eliminar una parte de los valores ausentes
mask = df1['days_employed'] > 0
df1.loc[mask,'occupation_type'] = df1.loc[mask,'occupation_type'].fillna('unployed')
df1.fillna('unknown',inplace=True)

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

# Se hace el mapeo para cambiar a valores enteros
df2["status_score"] = df2["status"].map(status_map)
df2['target'] = np.where(df2['status_score'] >= 2, 1, 0)

# Se combinan los dos datasets por el id y el valor target
merged = pd.merge(df1, df2[["id", "target"]], on="id", how="inner")

# Se separan en caracteristicas y objetivos
features = merged.drop(['id','target'],axis=1)
target = merged['target']

# Se seapara una seccion(10%) para realizar una comprobacion del modelo
features_train_1, features_retest, target_train_1, target_retest = train_test_split(
    features,
    target,
    test_size=0.1,
    random_state=12345
)

# Ahora se separan en entrenamiento y prueba
features_train, features_test, target_train, target_test = train_test_split(
    features_train_1,
    target_train_1,
    test_size=0.2,
    random_state=12345
)

# Se valancean los datos 
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

# Creo los parametros para el modelo
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

# Cambbio los valores de 'object' a 'categorico' para que el modelo lo pueda procesar
for col in columnas_texto:
    features_downsample_train[col] = features_downsample_train[col].astype('category')
    features_downsample_test[col] = features_downsample_test[col].astype('category')
    features_downsample_retest[col] = features_downsample_retest[col].astype('category')
    
# Para crear la validación cruzada de 5 folds 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

# Pongo a prueba el modelo con la validacion cruzada
for fold, (train_idx, val_idx) in enumerate(skf.split(features_downsample_train, target_downsample_train)):
    X_train_fold, X_val_fold = features_downsample_train.iloc[train_idx], features_downsample_train.iloc[val_idx]
    y_train_fold, y_val_fold = target_downsample_train.iloc[train_idx], target_downsample_train.iloc[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    test_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=100)]
    )

    pred_probs = model.predict(X_val_fold)
    auc_scores.append(roc_auc_score(y_val_fold, pred_probs))
    result = np.mean(auc_scores)
    
#retest_data = lgb.Dataset(features_downsample_retest, label=target_downsample_retest, reference=train_data)    
retest_predict = model.predict(features_downsample_retest)
validation_result = roc_auc_score(target_downsample_retest, retest_predict)
    
# Obtengo el valor de la validacion cruzada, el retesteo para ver el resultado original y el coeficiente gini
print(f"ROC-AUC Score: {result:.4f}")
print(f'ROC-AUC de segundo testeo:{validation_result:.4f}')
print(f"Coeficiente Gini:{(2*result)-1}")


# Empiezo a graficar los resultados
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Curva ROC
fpr, tpr, _ = roc_curve(y_val_fold, pred_probs)
roc_auc = auc(fpr, tpr)
ax[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
ax[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax[0].set_xlim([0.0, 1.0])
ax[0].set_ylim([0.0, 1.05])
ax[0].set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
ax[0].set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
ax[0].set_title('Curva ROC (Validación)')
ax[0].legend(loc="lower right")
ax[0].grid(True, alpha=0.3)

# Curva Precision-Recall
precision, recall, _ = precision_recall_curve(y_val_fold, pred_probs)
pr_auc = auc(recall, precision)
ax[1].plot(recall, precision, color='green', lw=2, label=f'PR curve (area = {pr_auc:.4f})')
ax[1].axhline(y=0.20, color='red', linestyle='--', label='Línea base aleatoria (20%)')
ax[1].set_xlim([0.0, 1.0])
ax[1].set_ylim([0.0, 1.05])
ax[1].set_xlabel('Recall (Sensibilidad)')
ax[1].set_ylabel('Precisión')
ax[1].set_title('Curva Precision-Recall')
ax[1].legend(loc="lower left")
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('metricas_rendimiento.png', dpi=300) # Se guarda para tu GitHub
plt.show()

# Segunda grafica
umbral_optimo = 0.4  # Ajusto el umbral a 0.4 para mejores predicciones financieras
pred_clases_ajustadas = (pred_probs >= umbral_optimo).astype(int)

cm = confusion_matrix(y_val_fold, pred_clases_ajustadas)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predice Bueno (0)', 'Predice Malo (1)'],
            yticklabels=['Real Bueno (0)', 'Real Malo (1)'])
plt.ylabel('Etiqueta Real')
plt.xlabel('Etiqueta Predicha')
plt.title(f'Matriz de Confusión (Umbral de Riesgo = {umbral_optimo})')
plt.tight_layout()
plt.savefig('matriz_confusion.png', dpi=300) # Se guarda para tu GitHub
plt.show()




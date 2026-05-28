\# 📊 Credit Scoring Model: Optimización del Riesgo de Impago mediante Gradient Boosting



Este repositorio contiene el desarrollo de un modelo predictivo de scoring de crédito diseñado para evaluar el riesgo de impago de solicitantes de productos financieros. El proyecto aborda de forma directa el desafío de trabajar con datos altamente desbalanceados y se enfoca en encontrar el equilibrio óptimo entre la rentabilidad del negocio y el control de pérdidas por cartera vencida.



\---



\## 🎯 El Problema de Negocio

Las instituciones financieras se enfrentan constantemente al riesgo de otorgar créditos a personas que caerán en situación de mora (incumplimiento de pago). En este escenario de datos, nos enfrentamos a dos retos principales:

1\. \*\*Desbalanceo Crítico:\*\* El dataset presenta una proporción de \*\*66:1\*\* (98.5% clientes "buenos" frente a un 1.5% de clientes "malos"). Un modelo estándar tendería a ignorar a la clase minoritaria.

2\. \*\*Asimetría de Costos:\*\* Financieramente, un \*\*Falso Negativo\*\* (aprobar un crédito a un cliente malo) es drásticamente más costoso para el banco que un \*\*Falso Positivo\*\* (rechazar por error a un cliente bueno).



\---



\## 🛠️ Metodología y Solución Técnica



\### 1. Preparación de Datos y Control de Fugas

\* \*\*Manejo de Variables Categóricas:\*\* Se transformaron nativamente las variables de texto (`code\_gender`, `name\_income\_type`, etc.) al tipo `category` de Pandas para un procesamiento de alta velocidad.

\* \*\*Eliminación de Target Leakage:\*\* Se aisló estrictamente la variable objetivo (`status\_score`) del set de entrenamiento para evitar el sobreajuste y asegurar la validez matemática del modelo.



\### 2. Estrategia contra el Desbalanceo (Proporción 4:1)

Se usaron funciones de balanceo para cambiar la proporción de 66:1 a 4:1.

Se utilizó \*\*LightGBM\*\* configurando el parámetro matemático exacto de penalización:

$$\\text{scale\\\_pos\\\_weight} = \\frac{\\text{Clientes Buenos}}{\\text{Clientes Malos}} = 4$$

Esto fuerza al algoritmo a penalizar 4 veces más los errores cometidos al clasificar a un cliente moroso.



\### 3. Validación Cruzada Robusta

Para garantizar la estabilidad del modelo fuera de muestra, se implementó \*\*Stratified K-Fold Cross-Validation\*\* (5 bloques), manteniendo de forma rigurosa la proporción 4:1 en cada iteración.



\---



\## 📈 Resultados y Validación de Negocio



El modelo alcanzó un rendimiento sobresaliente que supera el promedio estándar de la industria de riesgo crediticio:



\*   \*\*ROC-AUC Global:\*\* \*\*0.9081\*\* (Excelente capacidad para ordenar los perfiles de menor a mayor riesgo).

\*   \*\*Coeficiente de Gini:\*\* \*\*0.8162\*\* ($\\text{Gini} = 2 \\times \\text{ROC-AUC} - 1$). Un Gini > 0.80 clasifica al modelo en un nivel de desempeño \*Élite\*.



\### ⚖️ Optimización del Umbral de Decisión (Matriz de Confusión)

En lugar de utilizar el umbral por defecto de `0.5`, se realizó un análisis de curvas de decisión para calibrar la política de riesgo de la institución, encontrando el punto óptimo de rentabilidad:





| Clasificación | Cantidad de Clientes | Impacto Económico / Significado |

| :--- | :---: | :--- |

| \*\*Verdaderos Buenos (0,0)\*\* | \*\*10,096\*\* | Créditos otorgados sanos que generan intereses y utilidades. |

| \*\*Verdaderos Malos (1,1)\*\* | \*\*3,000\*\* | Riesgo mitigado de forma quirúrgica; pérdidas evitadas. |

| \*\*Falsos Malos (0,1)\*\* | \*\*2,444\*\* | Costo de oportunidad controlado (clientes buenos rechazados). |

| \*\*Falsos Buenos (1,0)\*\* | \*\*336\*\* | Tasa de escape de morosidad controlada a solo un \*\*10%\*\*. |



\*\*Conclusión Financiera:\*\* Al ajustar estratégicamente el umbral, se logró desbloquear la aprobación del \*\*80.5%\*\* de la base de clientes sanos, manteniendo la tasa de escape de morosos en un límite seguro del 10%. Los ingresos por intereses generados por los 10,096 clientes aprobados cubren con creces el costo del riesgo asumido.



\---



\## 🚀 Requisitos e Instalación

Para replicar el entrenamiento y generar las gráficas de validación, clona este repositorio e instala las dependencias:



```bash

git clone https://github.com

cd Credit\_Scoring

pip install -r requirements.txt

python src/CreditScoring.py

```




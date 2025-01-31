import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Adatok beolvasása
path = "C:/Users/Botond/PycharmProjects/californian-houseprices-machinelearning/Data/housing.csv/housing.csv"
data = pd.read_csv(path)

# Hiányzó adatok eltávolítása
data = data.dropna()

# Outlierek eltávolítása - Median House Value
Q1 = data['median_house_value'].quantile(0.25)
Q3 = data['median_house_value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 0.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['median_house_value'] >= lower_bound) & (data['median_house_value'] <= upper_bound)]

# Outlierek eltávolítása - Median Income
Q1 = data['median_income'].quantile(0.25)
Q3 = data['median_income'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['median_income'] >= lower_bound) & (data['median_income'] <= upper_bound)]

# Nem szükséges oszlop eltávolítása
data = data.drop(columns=["total_bedrooms"])

# One-hot encoding a kategóriákra
ocean_proximity_dummies = pd.get_dummies(data['ocean_proximity'], prefix='ocean_proximity', dtype=int)
data = pd.concat([data.drop("ocean_proximity", axis=1), ocean_proximity_dummies], axis=1)

# Opcionális: Ha az "ISLAND" kategória eltávolítása szükséges
if 'ocean_proximity_ISLAND' in data.columns:
    data = data.drop("ocean_proximity_ISLAND", axis=1)

# Feature és target változók definiálása
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'population', 'households', 'median_income',
            'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
            'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']
target = ["median_house_value"]

X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1111)

# Adattípusok konverziója biztosítva
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')
y_train = y_train.apply(pd.to_numeric, errors='coerce')
y_test = y_test.apply(pd.to_numeric, errors='coerce')

# Null értékek pótlása biztosítva
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Konstans oszlop hozzáadása a modellhez
X_train_const = sm.add_constant(X_train)

# OLS modell illesztése
model_fitted = sm.OLS(y_train, X_train_const).fit()

# Modell eredményeinek kiírása
print(model_fitted.summary())

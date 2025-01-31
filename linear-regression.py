import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

# Adatok beolvasása
path = "C:/Users/Botond/PycharmProjects/californian-houseprices-machinelearning/Data/housing.csv/housing.csv"
data = pd.read_csv(path)

# Hiányzó adatok eltávolítása
data = data.dropna()

# Outlierek eltávolítása - Median House Value
Q1 = data['median_house_value'].quantile(0.25)
Q3 = data['median_house_value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
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
# print(model_fitted.summary())

# trying to predict
X_test_const = sm.add_constant(X_test)

# Making predictions on the test set
test_predictions = model_fitted.predict(X_test_const)
# print(test_predictions) getting the predictions

# we can notice that we have a huge standard error which means that one of the requirements of the linear regression was broken
# we shall check all of them one-by-one
# first linearity
# Scatter plot for observed vs predicted values on test data
plt.scatter(y_test, test_predictions, color = "forestgreen")
plt.xlabel('Observed Values')
plt.ylabel('Predicted Values')
plt.title('Observed vs Predicted Values on Test Data')
plt.plot(y_test, y_test, color='darkred')  # line for perfect prediction (true values)
# plt.show()

# we then check if the sample is really random
# Calculate the mean of the residuals
mean_residuals = np.mean(model_fitted.resid)
# print(f"The mean of the residuals is {np.round(mean_residuals,2)}")
# another way to check the second requirement is to use a plot the following way:
# Plotting the residuals
plt.scatter(model_fitted.fittedvalues, model_fitted.resid, color = "forestgreen")
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
# plt.show()
# a harmadik requirement a exogeneity:
# Calculate the residuals
residuals = model_fitted.resid

# Check for correlation between residuals and each predictor
for column in X_train.columns:
    corr_coefficient = np.corrcoef(X_train[column], residuals)[0, 1]
    #print(f'Correlation between residuals and {column}: {np.round(corr_coefficient,2)}')
# standardizing the data
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same transformation to the test data
X_test_scaled = scaler.transform(X_test)
# Create and fit the model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
y_pred = lr.predict(X_test_scaled)

# Calculate MSE and RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)

# Output the performance metrics
print(f'RMSE on Test Set: {rmse}')
print(y_pred)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
path = "C:/Users/Botond/PycharmProjects/californian-houseprices-machinelearning/Data/housing.csv/housing.csv"
data = pd.read_csv(path)
#print(data["ocean_proximity"].unique())
missing_values = data.isnull().sum()
#print(missing_values)
missing_percentage = missing_values * 100 / len(data)
#print(missing_percentage)
cleaned_data = data.dropna()
#print(cleaned_data.isnull().sum())
#print(data.describe())
# visualizing house values

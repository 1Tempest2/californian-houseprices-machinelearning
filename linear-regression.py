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
#sns.set(style="whitegrid")
#plt.figure(figsize=(10, 6))
#sns.histplot(cleaned_data['median_house_value'], color='forestgreen', kde=True)
#plt.title('Distribution of Median House Values')
#plt.xlabel('Median House Value')
#plt.ylabel('Frequency')
#plt.show()

Q1 = cleaned_data['median_house_value'].quantile(0.25)
print("Q1:", Q1)
Q3 = cleaned_data['median_house_value'].quantile(0.75)
print("Q3", Q3)
IQR = Q3 - Q1
print("IQR:", IQR)


# Define the bounds for the outliers
lower_bound = Q1 - 0.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(lower_bound, upper_bound)
# Remove outliers
data_no_outliers_1 = cleaned_data[(cleaned_data['median_house_value'] >= lower_bound) & (cleaned_data['median_house_value'] <= upper_bound)]

# Check the shape of the data before and after removal of outliers
#print("Original data shape:", cleaned_data.shape)
#print("New data shape without outliers:", data_no_outliers_1.shape)
plt.figure(figsize=(10, 6))
sns.histplot(x=data_no_outliers_1['median_income'], color='purple')
plt.title('Outlier Analysis in Median Income')
plt.xlabel('Median Income')
#plt.show()
# Calculate Q1 and Q3
Q1 = data_no_outliers_1['median_income'].quantile(0.25)
Q3 = data_no_outliers_1['median_income'].quantile(0.75)
IQR = Q3 - Q1

# Define the bounds for the outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
data_no_outliers_2 = data_no_outliers_1[(data_no_outliers_1['median_income'] >= lower_bound) & (data_no_outliers_1['median_income'] <= upper_bound)]

# Check the shape of the data before and after the removal of outliers
#print("Original data shape:", data_no_outliers_1.shape)
#print("Data shape without outliers:", data_no_outliers_2.shape)
data = data_no_outliers_2 # just for simplicity
# heatmap
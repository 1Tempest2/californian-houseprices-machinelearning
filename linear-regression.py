import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
file_path = "C:/Users/Botond/PycharmProjects/californian-houseprices-machinelearning/Data/housing.csv/housing.csv"
data = pd.read_csv(file_path)
print(data.columns)
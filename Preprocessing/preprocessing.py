# Data Preprocess

import numpy as np      #mathametical
import matplotlib.pyplot as plt
import pandas as pd    #import datasets

#dataset
dataset = pd.read_csv('Data.csv')
X= dataset.iloc[:, :-1].values
Y= dataset.iloc[:, 3].values

# Missing Data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values= 'NaN',strategy ='mean', axis =0)
imputer = imputer.fit(X[:, 1:3]) 
X[:, 1:3]= imputer.transform(X[:, 1:3])
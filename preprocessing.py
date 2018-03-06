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

 # Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lablelencoder_X= LabelEncoder()
X[:, 0] = lablelencoder_X.fit_transform(X[:, 0])
onehotencoder= OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
lablelencoder_Y= LabelEncoder()
Y = lablelencoder_Y.fit_transform(Y)

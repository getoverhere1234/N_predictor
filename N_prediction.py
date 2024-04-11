#this uses sklearn

import pandas as pd
import numpy as np
import streamlit as st


#to load data set, put the csv file in root of pyscript
dataset = pd.read_csv('dataset.csv')
dataset.head(20)
#to view data
#dont use after frst use
print(dataset)

#entire data set i.e rows and column
dataset.shape
#to describe data
dataset.describe()

#to determine x(predictar) and y(dependant)
# X is captial, y is smaller letter
X = dataset.iloc[:, 1:10].values #this selects all rows from column 1 to 9
y = dataset.iloc[:, 0].values #first coloumn is zero in dataset

#i.e from sklearn we import train test split model, this sets data for testing
from sklearn.model_selection import train_test_split
#Test size is 20% meanwhile random_state=0 means data is split in a reproducible manner,
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


#standard scaler ensures data cleaning and scales it a standard form, this is usually done
#in values ranging from 0 to 1
from sklearn.preprocessing import StandardScaler
#assign standard scalar as sc
sc= StandardScaler()
#next code standardises xtrain and xtest
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#random forest regression is good for continous data, we selected random forest for this
from sklearn.ensemble import RandomForestRegressor

#we assign 20, split
regressor = RandomForestRegressor (n_estimators=20, random_state=0)

#train the data and predict
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

#evaluate the model
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#to check predicted and actual value
#df=pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#print(df)

from sklearn.metrics import confusion_matrix, accuracy_score
print("Training Accuracy = ", regressor.score(X_train, y_train))
print("Testing Accuracy = ", regressor.score(X_test, y_test))

#next run linear regression between actual and predicted values
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Save the scatter plot as an image file
plt.scatter(y_test, y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression between Actual and Predicted Values')
plt.savefig('scatter_plot.png')

# Display the saved image in the Streamlit app
st.image('scatter_plot.png', caption='Scatter plot between actual and predicted values')

st.title('Dependent Variable Prediction')

# Create input fields for independent variables
ph = st.number_input('pH', step=0.1, value=7.0)
ec = st.number_input('EC', step=0.1, value=0.0)
oc = st.number_input('OC', step=0.1, value=0.0)
S = st.number_input('S', step=0.1, value=0.0)
B = st.number_input('B', step=0.1, value=0.0)
Zn = st.number_input('Zn', step=0.1, value=0.0)
Fe = st.number_input('Fe', step=0.1, value=0.0)
Mn = st.number_input('Mn', step=0.1, value=0.0)
Cu = st.number_input('Cu', step=0.1, value=0.0)

# Transform inputs into a format suitable for prediction
input_data = np.array([[ph, ec, oc, S, B, Zn, Fe, Mn, Cu]])

# Load the trained model
regressor = RandomForestRegressor(n_estimators=20, random_state=0)  # You may need to load your trained model here

# Make prediction
# Assuming dataset is your training data used to train the model
# Assuming you have already applied StandardScaler to your training data
# You may need to adjust this part based on your actual implementation
dataset = pd.read_csv('dataset.csv')  # Load your dataset
X_train = dataset.iloc[:, 1:10].values  # Assuming your independent variables are in columns 1 to 9
y_train = dataset.iloc[:, 0].values  # Assuming your dependent variable is in the first column
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Scale your training data
regressor.fit(X_train_scaled, y_train)  # Train your model
input_data_scaled = scaler.transform(input_data)  # Scale the input data
prediction = regressor.predict(input_data_scaled)

# Display prediction
if st.button('Predict'):
    st.write('Predicted value of N:', prediction[0])

#after everything to use this in streamlit
#use this in terminal given below
#pip freeze> requirements.txt
#next enable version control integration from VCS
#


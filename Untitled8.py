#!/usr/bin/env python
# coding: utf-8

# # Import library for read the dataset

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("creditcard.csv")


# In[3]:


df


# #  Data Analysis
# 

# In[4]:


df.info()


# In[5]:


print(df.shape)


# In[6]:


print(df.describe)


# In[7]:


df.count()


# In[8]:


df.isnull()


# In[9]:


df.isnull().sum()


# In[10]:


print(df.columns)


# In[11]:


filtered_data = df[(df['V1'] >= -1.359807) & (df['V1'] <= 1.058415)]

print(filtered_data)


# In[16]:


filtered_data = df[(df['Amount'] >= 9) & (df['Amount'] <= 10)]

print(filtered_data)


# # Data Visualization

# In[17]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[18]:


df.hist(bins=10, figsize=(10, 8))
plt.tight_layout()
plt.show()


# In[22]:


# Define the colors based on conditions
colors = ['red' if length >=-1.029719 else 'yellow' for length in df['V1']]

# Scatter plot with different colors
plt.scatter(df['V1'], df['Amount'], c=colors)
plt.xlabel('V1')
plt.ylabel('Amount')
plt.title('V1 vs Amount')

# Show the plot
plt.show()


# In[23]:


# Define the colors based on conditions
colors = ['red' if length >= 0.754195 else 'yellow' for length in df['V21']]

# Scatter plot with different colors
plt.scatter(df['V21'], df['Amount'], c=colors)
plt.xlabel('V21')
plt.ylabel('Amount')
plt.title('V21 vs Amount')

# Show the plot
plt.show()


# In[25]:


# Select the columns for correlation
columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'Amount']

# Calculate the coefficient matrix
correlation_matrix = df[columns].corr()

# Display the coefficient matrix
print(correlation_matrix)

# Plot the correlation matrix as a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[28]:


# Select the columns for correlation
columns = ['Time', 'V8', 'V9', 'V10','V11', 'V12', 'V13', 'V14', 'Amount']

# Calculate the coefficient matrix
correlation_matrix = df[columns].corr()

# Display the coefficient matrix
print(correlation_matrix)

# Plot the correlation matrix as a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[29]:


# Select the columns for correlation
columns = ['Time', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'Amount']

# Calculate the coefficient matrix
correlation_matrix = df[columns].corr()

# Display the coefficient matrix
print(correlation_matrix)

# Plot the correlation matrix as a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[30]:


# Select the columns for correlation
columns = ['Time', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
       'Class']

# Calculate the coefficient matrix
correlation_matrix = df[columns].corr()

# Display the coefficient matrix
print(correlation_matrix)

# Plot the correlation matrix as a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# # Machine Learning
# # Linear Regression Models (Using first 14 Columns) :-

# In[36]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Assuming your dataset is stored in a DataFrame named 'df'

# Splitting the data into features (X) and target variable (y)
X = df[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'Amount'  ]]
y = df['Amount']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predicting on the test set
y_pred = lr.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Calculating accuracy
threshold = 0.1  # Define your threshold value here
accurate_predictions = (abs(y_test - y_pred) <= threshold).sum()
total_predictions = len(y_test)
accuracy = accurate_predictions / total_predictions

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)
print("Accuracy:", accuracy)


# # Linear Regression Models (Using 2nd 14 Columns) :-

# In[37]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Assuming your dataset is stored in a DataFrame named 'df'

# Splitting the data into features (X) and target variable (y)
X = df[['Time', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'  ]]
y = df['Amount']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predicting on the test set
y_pred = lr.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Calculating accuracy
threshold = 0.1  # Define your threshold value here
accurate_predictions = (abs(y_test - y_pred) <= threshold).sum()
total_predictions = len(y_test)
accuracy = accurate_predictions / total_predictions

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)
print("Accuracy:", accuracy)


# 

# In[ ]:





# In[ ]:





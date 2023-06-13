#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
 
data= pd.read_csv("2021-2022.csv")
data


# In[2]:


data= data.drop(['Div','Date','Time','Referee','AHh','AHCh'],axis=1)
data


# In[3]:


data['HomeTeam'] = data['HomeTeam'].map({'Charlton':1,'Chelsea':2,'Coventry':3, 'Derby':4, 'Leeds':5, 'Leicester':6,'Liverpool':7, 'Sunderland':8, 'Tottenham':9, 'Man United':10, 'Arsenal':11,'Bradford':12, 'Ipswich':13, 'Middlesbrough':14, 'Everton':15, 'Man City':16,'Newcastle':17, 'Southampton':18, 'West Ham':19, 'Aston Villa':20, 'Bolton':21,
       'Blackburn':22, 'Fulham':23, 'Birmingham':24, 'Middlesboro':25, 'West Brom':26,
       'Portsmouth':27, 'Wolves':28, 'Norwich':29, 'Crystal Palace':30, 'Wigan':31,
       'Reading':32, 'Sheffield United':33, 'Watford':34, 'Hull':35, 'Stoke':36,
       'Burnley':37, 'Blackpool':38, 'QPR':39, 'Swansea':40, 'Cardiff':41, 'Bournemouth':42,
       'Brighton':43, 'Huddersfield':44, 'Brentford':45}).astype('int')


# In[4]:


data['AwayTeam'] = data['AwayTeam'].map({'Man City':1, 'West Ham':2, 'Middlesbrough':3, 'Southampton':4, 'Everton':5,
       'Aston Villa':6, 'Bradford':7, 'Arsenal':8, 'Ipswich':9, 'Newcastle':10,
       'Liverpool':11, 'Chelsea':12, 'Man United':13, 'Tottenham':14, 'Charlton':15,
       'Sunderland':16, 'Derby':17, 'Coventry':18, 'Leicester':19, 'Leeds':20,
       'Blackburn':21, 'Bolton':22, 'Fulham':23, 'West Brom':24, 'Middlesboro':25,
       'Birmingham':26, 'Wolves':27, 'Portsmouth':28, 'Crystal Palace':29, 'Norwich':30,
       'Wigan':31, 'Watford':32, 'Sheffield United':33, 'Reading':34, 'Stoke':35, 'Hull':36,
       'Burnley':37, 'Blackpool':38, 'Swansea':39, 'QPR':40, 'Cardiff':41, 'Bournemouth':42,
       'Huddersfield':43, 'Brighton':44, 'Brentford':45}).astype('int')


# In[5]:


data['FTR'] = data['FTR'].map({'H':2,'A':1,'D':0}).astype('int')
data['FTR'].unique()


# In[6]:


data['HTR'] = data['HTR'].map({'H':2,'A':1,'D':0}).astype('int')
data['HTR'].unique()


# In[7]:


y = data['FTR']
#Load X Variables into a Pandas Dataframe with columns 
X = data.drop(['FTR'], axis = 1)
#y is dependent variable and X is independent variable.


# In[8]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=100,test_size=0.3) 


# In[9]:


cor = X_train.corr()
plt.figure(figsize=(22,20))
sns.heatmap(cor, cmap=plt.cm.CMRmap_r,annot=True)
plt.show() 


# In[10]:


def correlation(dataset, threshold):
    col_corr = set()  
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i] 
                col_corr.add(colname)
    return col_corr 


# In[11]:


corr_features = correlation(X_train, 0.7)
corr_features


# In[12]:


X_train.drop(corr_features,axis=1)
X_test.drop(corr_features,axis=1)
X_train 


# In[13]:


#Chi-square
#Perform chi2 test
from sklearn.feature_selection import chi2
#Calculating Fscore and p value
f_p_values=chi2(X_train,y_train)
f_p_values


# In[14]:


import pandas as pd
p_values=pd.Series(f_p_values[1])
p_values.index=X_train.columns
p_values 


# In[15]:


p_values.sort_index(ascending=False)


# In[16]:


#Mutual information gain
#Importing mutual information gain
from sklearn.feature_selection import mutual_info_classif
# determine the mutual information
mutual_info = mutual_info_classif(X_train, y_train)
mutual_info


# In[17]:


#Representing in list form
mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info.sort_values(ascending=False)


# In[18]:


mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 8))


# In[19]:


from sklearn.feature_selection import SelectKBest
#No we Will select the top 10 important features
sel_five_cols = SelectKBest(mutual_info_classif, k=10)
sel_five_cols.fit(X_train, y_train)
X_train.columns[sel_five_cols.get_support()]


# In[20]:


columns = ['HomeTeam', 'AwayTeam', 'AST','HST','HS','AS']
X = data[columns]
Y = data['FTR']


# In[21]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)


# In[22]:


import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# # LOGISTIC REGRESSION

# In[23]:


modelLogisticRegression = LogisticRegression()
modelLogisticRegression.fit(X_train, Y_train)
X_train_prediction = modelLogisticRegression.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Training Accuracy:",round(training_data_accuracy*100,2),'%')
X_test_prediction = modelLogisticRegression.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Testing Accuracy:",round(test_data_accuracy*100,2),'%')


# # Logistic Regression Hyperparameter tuning

# In[33]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l2']
}

# Create the Logistic Regression model
modelLogisticRegression = LogisticRegression(max_iter=1000,penalty='l1', solver='liblinear')

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(estimator=modelLogisticRegression, param_grid=param_grid, cv=5)
grid_search.fit(X_train, Y_train)

# Get the best model and its hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Print the best hyperparameters
print("Best Hyperparameters:", best_params)

# Evaluate the best model on the training data
X_train_prediction = best_model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Training Accuracy:", round(training_data_accuracy * 100, 2), "%")

# Evaluate the best model on the test data
X_test_prediction = best_model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Testing Accuracy:", round(test_data_accuracy * 100, 2), "%")


# # SVC

# In[24]:


from sklearn.calibration import CalibratedClassifierCV
modelSVC = SVC()
modelSVC = CalibratedClassifierCV(modelSVC)
modelSVC.fit(X_train, Y_train)
X_train_prediction = modelSVC.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Training Accuracy:",round(training_data_accuracy*100,2),'%')

X_test_prediction = modelSVC.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Testing Accuracy:",round(test_data_accuracy*100,2),'%')
SVCTEST = round(test_data_accuracy*100,2)


# # XGBOOST

# In[25]:


modelXGBClassifier = xgb.XGBClassifier()
modelXGBClassifier.fit(X_train, Y_train)
X_train_prediction = modelXGBClassifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Traning Accuracy:",round(training_data_accuracy*100,2),'%')

X_test_prediction = modelXGBClassifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Test Accuracy:",round(test_data_accuracy*100,2),'%')
XGBTEST = round(test_data_accuracy*100,2)


# # Xgboost Hyperparameter tuning

# In[34]:


import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300]
}

# Create the XGBoost classifier
modelXGBClassifier = xgb.XGBClassifier()

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(estimator=modelXGBClassifier, param_grid=param_grid, cv=5)
grid_search.fit(X_train, Y_train)

# Get the best model and its hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Print the best hyperparameters
print("Best Hyperparameters:", best_params)

# Evaluate the best model on the training data
X_train_prediction = best_model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Training Accuracy:", round(training_data_accuracy * 100, 2), "%")

# Evaluate the best model on the test data
X_test_prediction = best_model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Testing Accuracy:", round(test_data_accuracy * 100, 2), "%")

# Save the accuracy for later use
XGBTEST = round(test_data_accuracy * 100, 2)


# # GAUSSIAN NAIVE BAYES

# In[26]:


from sklearn.naive_bayes import GaussianNB
modelGaussianNB = GaussianNB()
modelGaussianNB.fit(X_train, Y_train)
X_train_prediction = modelGaussianNB.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Training Accuracy:",round(training_data_accuracy*100,2),'%')
X_test_prediction = modelGaussianNB.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Testing Accuracy:",round(test_data_accuracy*100,2),'%')
SVCTEST = round(test_data_accuracy*100,2)


# # Gaussian Naive Bayes Hyper parameter tuning

# In[36]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Create the Gaussian Naive Bayes model
modelGaussianNB = GaussianNB()

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7]
}

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(estimator=modelGaussianNB, param_grid=param_grid, cv=5)
grid_search.fit(X_train, Y_train)

# Get the best model and its hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Print the best hyperparameters
print("Best Hyperparameters:", best_params)

# Evaluate the best model on the training data
X_train_prediction = best_model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Training Accuracy:", round(training_data_accuracy * 100, 2), "%")

# Evaluate the best model on the test data
X_test_prediction = best_model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Testing Accuracy:", round(test_data_accuracy * 100, 2), "%")

# Save the accuracy for later use
SVCTEST = round(test_data_accuracy * 100, 2)


# # RANDOM FOREST CLASSIFIER

# In[27]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)
X_train_prediction = rf.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Training Accuracy:",round(training_data_accuracy*100,2),'%')
X_test_prediction = rf.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Testing Accuracy:",round(test_data_accuracy*100,2),'%')


# # Random Forest Classifier (Hyperparameter tuning)

# In[31]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# Create the random forest classifier
rf = RandomForestClassifier(random_state=42)

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, Y_train)

# Get the best model and its hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Print the best hyperparameters
print("Best Hyperparameters:", best_params)

# Evaluate the best model on the training data
X_train_prediction = best_model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Training Accuracy:", round(training_data_accuracy * 100, 2), "%")

# Evaluate the best model on the test data
X_test_prediction = best_model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Testing Accuracy:", round(test_data_accuracy * 100, 2), "%")


# In[ ]:





import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from IPython.display import display
from flask import Flask,render_template, request
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

app = Flask(__name__)

data = pd.read_csv('2021-2022.csv')
data= data.drop(['Div','Date','Time','Referee','AHh','AHCh'],axis=1)

data['HomeTeam'] = data['HomeTeam'].map({'Charlton':1,'Chelsea':2,'Coventry':3, 'Derby':4, 'Leeds':5, 'Leicester':6,'Liverpool':7, 'Sunderland':8, 'Tottenham':9, 'Man United':10, 'Arsenal':11,'Bradford':12, 'Ipswich':13, 'Middlesbrough':14, 'Everton':15, 'Man City':16,'Newcastle':17, 'Southampton':18, 'West Ham':19, 'Aston Villa':20, 'Bolton':21,
       'Blackburn':22, 'Fulham':23, 'Birmingham':24, 'Middlesboro':25, 'West Brom':26,
       'Portsmouth':27, 'Wolves':28, 'Norwich':29, 'Crystal Palace':30, 'Wigan':31,
       'Reading':32, 'Sheffield United':33, 'Watford':34, 'Hull':35, 'Stoke':36,
       'Burnley':37, 'Blackpool':38, 'QPR':39, 'Swansea':40, 'Cardiff':41, 'Bournemouth':42,
       'Brighton':43, 'Huddersfield':44, 'Brentford':45}).astype('int')

data['AwayTeam'] = data['AwayTeam'].map({'Man City':1, 'West Ham':2, 'Middlesbrough':3, 'Southampton':4, 'Everton':5,
       'Aston Villa':6, 'Bradford':7, 'Arsenal':8, 'Ipswich':9, 'Newcastle':10,
       'Liverpool':11, 'Chelsea':12, 'Man United':13, 'Tottenham':14, 'Charlton':15,
       'Sunderland':16, 'Derby':17, 'Coventry':18, 'Leicester':19, 'Leeds':20,
       'Blackburn':21, 'Bolton':22, 'Fulham':23, 'West Brom':24, 'Middlesboro':25,
       'Birmingham':26, 'Wolves':27, 'Portsmouth':28, 'Crystal Palace':29, 'Norwich':30,
       'Wigan':31, 'Watford':32, 'Sheffield United':33, 'Reading':34, 'Stoke':35, 'Hull':36,
       'Burnley':37, 'Blackpool':38, 'Swansea':39, 'QPR':40, 'Cardiff':41, 'Bournemouth':42,
       'Huddersfield':43, 'Brighton':44, 'Brentford':45}).astype('int')

data['FTR'] = data['FTR'].map({'H':2,'A':1,'D':0}).astype('int')
data['FTR'].unique()

data1 = data.drop(['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HTR'], axis=1)
data = data1

columns = ['HomeTeam', 'AwayTeam', 'AST','HST','HS','AS']
X = data[columns]
Y = data['FTR']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

"""modelLogisticRegression = LogisticRegression(multi_class='multinomial', solver='lbfgs')
modelLogisticRegression.fit(X_train, Y_train)
X_train_prediction = modelLogisticRegression.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Training Accuracy:",round(training_data_accuracy*100,2),'%')
X_test_prediction = modelLogisticRegression.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Testing Accuracy:",round(test_data_accuracy*100,2),'%')"""


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


rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, Y_train)
X_train_prediction = rf.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Training Accuracy:", round(training_data_accuracy * 100, 2), '%')
X_test_prediction = rf.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Testing Accuracy:", round(test_data_accuracy * 100, 2), '%')





@app.route('//')
def homepage():
       return render_template('HomePage.html')

@app.route('/home')
def homepage1():
       return render_template('HomePage.html')

@app.route('/intro')
def intro():
    return render_template("IntroPage.html")

@app.route("/", methods=['POST','GET'])
def intropage():
       if request.method == 'POST':

              home = int(request.form.get('HomeTeam'))
              away = int(request.form.get('AwayTeam'))

              input_data = (home, away, 0, 0, 0, 0)

              #Logistic Regression
              """input_data_as_numpy_array = np.asarray(input_data)
              input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
              prediction = modelLogisticRegression.predict_proba(input_data_reshaped)
              print(prediction)
              arr = np.array(prediction)
              arr_perc = np.round(arr*100, 2)
              arr_no_brackets = np.squeeze(arr_perc)"""

              #XGBOOST
              input_data_as_numpy_array = np.asarray(input_data)
              input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
              prediction = grid_search.predict_proba(input_data_reshaped)
              print(prediction)
              arr = np.array(prediction)
              arr_perc = np.round(arr * 100, 2)
              arr_no_brackets = np.squeeze(arr_perc)

              #Random Forest
              input_data_as_numpy_array1 = np.asarray(input_data)
              input_data_reshaped1 = input_data_as_numpy_array1.reshape(1, -1)
              prediction1 = rf.predict_proba(input_data_reshaped1)
              print(prediction1)
              arr1 = np.array(prediction1)
              arr_perc1 = np.round(arr1 * 100, 2)
              arr_no_brackets1 = np.squeeze(arr_perc1)

              if home == 1:
                     home = 'Charlton'

              elif home == 2:
                     home = 'Chelsea'

              elif home == 3:
                     home = 'Coventry'

              elif home == 4:
                     home = 'Derby'

              elif home == 5:
                     home = 'Leeds'

              elif home == 6:
                     home = 'Leicester'

              elif home == 7:
                     home = 'Liverpool'

              elif home == 8:
                     home = 'Sunderland'

              elif home == 9:
                     home = 'Tottenham'

              elif home == 10:
                     home = 'Man United'

              elif home == 11:
                     home = 'Arsenal'

              elif home == 12:
                     home = 'Bradford'

              elif home == 13:
                     home = 'Ipswich'

              elif home == 14:
                     home = 'Middlesbrough'

              elif home == 15:
                     home = 'Everton'

              elif home == 16:
                     home = 'Man City'

              elif home == 17:
                     home = 'Newcastle'

              elif home == 18:
                     home = 'Southampton'

              elif home == 19:
                     home = 'West Ham'

              elif home == 20:
                     home = 'Aston Villa'

              elif home == 21:
                     home = 'Bolton'

              elif home == 22:
                     home = 'Blackburn'

              elif home == 23:
                     home = 'Fulham'

              elif home == 24:
                     home = 'Birmingham'

              elif home == 25:
                     home = 'Middlesboro'

              elif home == 26:
                     home = 'West Brom'

              elif home == 27:
                     home = 'Portsmouth'

              elif home == 28:
                     home = 'Wolves'

              elif home == 29:
                     home = 'Norwich'

              elif home == 30:
                     home = 'Crystal Palace'

              elif home == 31:
                     home = 'Wigan'

              elif home == 32:
                     home = 'Reading'

              elif home == 33:
                     home = 'Sheffield United'

              elif home == 34:
                     home = 'Watford'

              elif home == 35:
                     home = 'Hull'

              elif home == 36:
                     home = 'Stoke'

              elif home == 37:
                     home = 'Burnley'

              elif home == 38:
                     home = 'Blackpool'

              elif home == 39:
                     home = 'QPR'

              elif home == 40:
                     home = 'Swansea'

              elif home == 41:
                     home = 'Cardiff'

              elif home == 42:
                     home = 'Bournemouth'

              elif home == 43:
                     home = 'Brighton'

              elif home == 44:
                     home = 'Huddersfield'

              elif home == 45:
                     home = 'Brentford'

              else:
                     print('none')

              #Away Team
              if away == 1:
                     away = 'Man City'

              elif away == 2:
                     away = 'West Ham'

              elif away == 3:
                     away = 'Middlesbrough'

              elif away == 4:
                     away = 'Southampton'

              elif away == 5:
                     away = 'Everton'

              elif away == 6:
                     away = 'Aston Villa'

              elif away == 7:
                     away = 'Bradford'

              elif away == 8:
                     away = 'Arsenal'

              elif away == 9:
                     away = 'Ipswich'

              elif away == 10:
                     away = 'Newcastle'

              elif away == 11:
                     away = 'Liverpool'

              elif away == 12:
                     away = 'Chelsea'

              elif away == 13:
                     away = 'Man United'

              elif away == 14:
                     away = 'Tottenham'

              elif away == 15:
                     away = 'Charlton'

              elif away == 16:
                     away = 'Sunderland'

              elif away == 17:
                     away = 'Derby'

              elif away == 18:
                     away = 'Coventry'

              elif away == 19:
                     away = 'Leicester'

              elif away == 20:
                     away = 'Leeds'

              elif away == 21:
                     away = 'Blackburn'

              elif away == 22:
                     away = 'Bolton'

              elif away == 23:
                     away = 'Fulham'

              elif away == 24:
                     away = 'West Brom'

              elif away == 25:
                     away = 'Middlesboro'

              elif away == 26:
                     away = 'Birmingham'

              elif away == 27:
                     away = 'Wolves'

              elif away == 28:
                     away = 'Portsmouth'

              elif away == 29:
                     away = 'Crystal Palace'

              elif away == 30:
                     away = 'Norwich'

              elif away == 31:
                     away = 'Wigan'

              elif away == 32:
                     away = 'Watford'

              elif away == 33:
                     away = 'Sheffield United'

              elif away == 34:
                     away = 'Reading'

              elif away == 35:
                     away = 'Stoke'

              elif away == 36:
                     away = 'Hull'

              elif away == 37:
                     away = 'Burnley'

              elif away == 38:
                     away = 'Blackpool'

              elif away == 39:
                     away = 'Swansea'

              elif away == 40:
                     away = 'QPR'

              elif away == 41:
                     away = 'Cardiff'

              elif away == 42:
                     away = 'Bournemouth'

              elif away == 43:
                     away = 'Huddersfield'

              elif away == 44:
                     away = 'Brighton'

              elif away == 45:
                     away = 'Brentford'

              else:
                     print('none')

              return render_template('IntroPage.html', home=home, away=away, pr=arr_no_brackets, pr1=arr_no_brackets1)

       else:
           return render_template('IntroPage.html')


@app.route('/second')
def second():
    return render_template("SecondPage.html")


@app.route("/he",methods=['GET','POST'])
def secondpage():
       if request.method == 'POST':

              home = int(request.form.get('HomeTeam'))
              away = int(request.form.get('AwayTeam'))
              Ast = int(request.form.get('Ast'))
              Hst = int(request.form.get('Hst'))
              Hs = int(request.form.get('Hs'))
              As = int(request.form.get('As'))

              input_data = (home, away, Ast, Hst, Hs, As)

              """# Logistic Regression
              input_data_as_numpy_array = np.asarray(input_data)
              input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
              prediction = modelLogisticRegression.predict_proba(input_data_reshaped)
              print(prediction)
              arr = np.array(prediction)
              arr_perc = np.round(arr * 100, 2)
              arr_no_brackets = np.squeeze(arr_perc)"""

              #XGBOOST
              input_data_as_numpy_array = np.asarray(input_data)
              input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
              prediction = grid_search.predict_proba(input_data_reshaped)
              print(prediction)
              arr = np.array(prediction)
              arr_perc = np.round(arr * 100, 2)
              arr_no_brackets = np.squeeze(arr_perc)

              # Random Forest
              input_data_as_numpy_array1 = np.asarray(input_data)
              input_data_reshaped1 = input_data_as_numpy_array1.reshape(1, -1)
              prediction1 = rf.predict_proba(input_data_reshaped1)
              print(prediction1)
              arr1 = np.array(prediction1)
              arr_perc1 = np.round(arr1 * 100, 2)
              arr_no_brackets1 = np.squeeze(arr_perc1)

              if home == 1:
                     home = 'Charlton'

              elif home == 2:
                     home = 'Chelsea'

              elif home == 3:
                     home = 'Coventry'

              elif home == 4:
                     home = 'Derby'

              elif home == 5:
                     home = 'Leeds'

              elif home == 6:
                     home = 'Leicester'

              elif home == 7:
                     home = 'Liverpool'

              elif home == 8:
                     home = 'Sunderland'

              elif home == 9:
                     home = 'Tottenham'

              elif home == 10:
                     home = 'Man United'

              elif home == 11:
                     home = 'Arsenal'

              elif home == 12:
                     home = 'Bradford'

              elif home == 13:
                     home = 'Ipswich'

              elif home == 14:
                     home = 'Middlesbrough'

              elif home == 15:
                     home = 'Everton'

              elif home == 16:
                     home = 'Man City'

              elif home == 17:
                     home = 'Newcastle'

              elif home == 18:
                     home = 'Southampton'

              elif home == 19:
                     home = 'West Ham'

              elif home == 20:
                     home = 'Aston Villa'

              elif home == 21:
                     home = 'Bolton'

              elif home == 22:
                     home = 'Blackburn'

              elif home == 23:
                     home = 'Fulham'

              elif home == 24:
                     home = 'Birmingham'

              elif home == 25:
                     home = 'Middlesboro'

              elif home == 26:
                     home = 'West Brom'

              elif home == 27:
                     home = 'Portsmouth'

              elif home == 28:
                     home = 'Wolves'

              elif home == 29:
                     home = 'Norwich'

              elif home == 30:
                     home = 'Crystal Palace'

              elif home == 31:
                     home = 'Wigan'

              elif home == 32:
                     home = 'Reading'

              elif home == 33:
                     home = 'Sheffield United'

              elif home == 34:
                     home = 'Watford'

              elif home == 35:
                     home = 'Hull'

              elif home == 36:
                     home = 'Stoke'

              elif home == 37:
                     home = 'Burnley'

              elif home == 38:
                     home = 'Blackpool'

              elif home == 39:
                     home = 'QPR'

              elif home == 40:
                     home = 'Swansea'

              elif home == 41:
                     home = 'Cardiff'

              elif home == 42:
                     home = 'Bournemouth'

              elif home == 43:
                     home = 'Brighton'

              elif home == 44:
                     home = 'Huddersfield'

              elif home == 45:
                     home = 'Brentford'

              else:
                     print('none')

              # Away Team
              if away == 1:
                     away = 'Man City'

              elif away == 2:
                     away = 'West Ham'

              elif away == 3:
                     away = 'Middlesbrough'

              elif away == 4:
                     away = 'Southampton'

              elif away == 5:
                     away = 'Everton'

              elif away == 6:
                     away = 'Aston Villa'

              elif away == 7:
                     away = 'Bradford'

              elif away == 8:
                     away = 'Arsenal'

              elif away == 9:
                     away = 'Ipswich'

              elif away == 10:
                     away = 'Newcastle'

              elif away == 11:
                     away = 'Liverpool'

              elif away == 12:
                     away = 'Chelsea'

              elif away == 13:
                     away = 'Man United'

              elif away == 14:
                     away = 'Tottenham'

              elif away == 15:
                     away = 'Charlton'

              elif away == 16:
                     away = 'Sunderland'

              elif away == 17:
                     away = 'Derby'

              elif away == 18:
                     away = 'Coventry'

              elif away == 19:
                     away = 'Leicester'

              elif away == 20:
                     away = 'Leeds'

              elif away == 21:
                     away = 'Blackburn'

              elif away == 22:
                     away = 'Bolton'

              elif away == 23:
                     away = 'Fulham'

              elif away == 24:
                     away = 'West Brom'

              elif away == 25:
                     away = 'Middlesboro'

              elif away == 26:
                     away = 'Birmingham'

              elif away == 27:
                     away = 'Wolves'

              elif away == 28:
                     away = 'Portsmouth'

              elif away == 29:
                     away = 'Crystal Palace'

              elif away == 30:
                     away = 'Norwich'

              elif away == 31:
                     away = 'Wigan'

              elif away == 32:
                     away = 'Watford'

              elif away == 33:
                     away = 'Sheffield United'

              elif away == 34:
                     away = 'Reading'

              elif away == 35:
                     away = 'Stoke'

              elif away == 36:
                     away = 'Hull'

              elif away == 37:
                     away = 'Burnley'

              elif away == 38:
                     away = 'Blackpool'

              elif away == 39:
                     away = 'Swansea'

              elif away == 40:
                     away = 'QPR'

              elif away == 41:
                     away = 'Cardiff'

              elif away == 42:
                     away = 'Bournemouth'

              elif away == 43:
                     away = 'Huddersfield'

              elif away == 44:
                     away = 'Brighton'

              elif away == 45:
                     away = 'Brentford'

              else:
                     print('none')

              return render_template('SecondPage.html', home=home, away=away, pr=arr_no_brackets, pr1=arr_no_brackets1)

       else:
              return render_template("SecondPage.html")


if __name__ == "__main__":
       app.run(debug=True)

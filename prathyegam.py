import numpy as np
import os
from numpy import loadtxt
import xgboost
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier  
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pickle
from flask import Flask, render_template,request

dataset = pd.read_csv('train.csv')
dataset.drop('Loan_ID', axis=1, inplace=True)
dataset.head(5)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cat = dataset.select_dtypes(include='O').keys()
cat = list(cat)
for i in cat:
  dataset[i] = le.fit_transform(dataset[i])
for i in dataset.columns:
  dataset[i].fillna(int(dataset[i].mean()), inplace=True)
dataset.to_csv('file2.csv', header=False, index=False)

dataset = loadtxt('file2.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:11]
Y = dataset[:,11]
# CV model
kfold = KFold(n_splits=10)
kfoldn = KFold(n_splits=400)

xgb_reg = xgboost.XGBRegressor(colsample_bytree= 0.6, gamma= 0.5, max_depth= 3, min_child_weight= 1, n_estimators= 180, reg_alpha= 40, reg_lambda= 0, seed= 0, subsample= 0.6)
xgb_reg.fit(X, Y)

abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion= "gini",max_depth= 2, min_samples_leaf= 5, splitter= "random"), learning_rate= 0.01, n_estimators= 10)
abc.fit(X, Y)

pickle.dump(abc, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1, 0, 0, 0, 0, 3500, 1500, 150, 360, 1, 0]]))

app=Flask(__name__,template_folder='templates')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    if output == 1.0:
        finout = "Approved"
    else:
        finout = "Not Approved"
    return render_template('form.html', prediction_text='Loan is :{}'.format(finout))

if __name__ == "__main__":
    app.run(debug=True)
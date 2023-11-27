from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
def home(request):
    return render(request, 'home.html')
def predict(request):
    return render(request, 'predict.html')
def result(request):
    dataset = pd.read_csv('/Users/songatejohn7/Desktop/diab_proj/updated_diabetes.csv')

    dataset_new = dataset
    # Replacing zero values with NaN
    dataset_new[["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi", "HbA1c_level",
                 "blood_glucose_level"]] = dataset_new[
        ["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi", "HbA1c_level",
         "blood_glucose_level"]].replace(0, np.NaN)

    # Replacing NaN with mean values
    dataset_new["gender"].fillna(dataset_new["gender"].mean(), inplace=True)
    dataset_new["age"].fillna(dataset_new["age"].mean(), inplace=True)
    dataset_new["hypertension"].fillna(dataset_new["hypertension"].mean(), inplace=True)
    dataset_new["heart_disease"].fillna(dataset_new["heart_disease"].mean(), inplace=True)
    dataset_new["smoking_history"].fillna(dataset_new["smoking_history"].mean(), inplace=True)
    dataset_new["bmi"].fillna(dataset_new["bmi"].mean(), inplace=True)
    dataset_new["HbA1c_level"].fillna(dataset_new["HbA1c_level"].mean(), inplace=True)
    dataset_new["blood_glucose_level"].fillna(dataset_new["blood_glucose_level"].mean(), inplace=True)

    X = dataset.drop('diabetes', axis=1)
    Y = dataset['diabetes']
    # Splitting X and Y
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    # Random forest Algorithm

    model = RandomForestClassifier(n_estimators=11, criterion='entropy', random_state=42)
    model.fit(X_train, Y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

    result1 = " "
    if pred == [1]:
        result1 = "Positive!!"
    else:
        result1 = "Negative!"


    return render(request, "predict.html",{"result2":result1} )
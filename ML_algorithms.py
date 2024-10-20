import numpy as np
import pandas as pd
import joblib  # Para guardar el modelo
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, Binarizer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from decouple import config
#from sklearn import *

def loadDT(algorithm, configuration, pathTraining, pathTest, laboratoryID, spliteType):
    # Cargar los conjuntos de datos divididos.
    X_train = pd.read_csv(pathTraining)
    X_test = pd.read_csv(pathTest)

    print(X_train)
    print(X_test)

    # Extraer la columna de target
    y_train = X_train['target']             #target de entrenamiento
    y_test = X_test['target']               #target de prueba
    #y_train = X_train.iloc[:, 5]
    #y_test = X_test.iloc[:, 8]

    # Eliminar la columna target de los conjuntos de features
    X_train = X_train[['start_position', 'end_position', 'chou_fasman', 'emini', 'kolaskar_tongaonkar', 'parker', 'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability']]  #features de entrenamiento
    X_test = X_test[['start_position', 'end_position', 'chou_fasman', 'emini', 'kolaskar_tongaonkar', 'parker', 'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability']]   #features de testeo
    #X_train = X_train.iloc[:, 0:2,4:7] ##Funcion para excluir una columna.
    #X_test = X_test.iloc[:, 0:8]
    X_train.shape, X_test.shape, y_train.shape, y_test.shape

    print(X_train)
    print(X_test)

    model_name = f"{algorithm}_{configuration}"
    funcion = globals()[model_name]
    accuracy, precision, recall, matriz_confusion, path_model = funcion(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType)

    return [accuracy, precision, recall, matriz_confusion, path_model, pathTraining, pathTest, model_name]

def evaluateAndSaveModel(X_test, y_test, model, model_name, laboratoryID, spliteType):
    # Evaluar el modelo con metricas
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    accuracy = round(accuracy, 2)
    precision = precision_score(y_test, y_pred) * 100
    precision = round(precision, 2)
    recall = recall_score(y_test, y_pred) * 100
    recall = round(recall, 2)

    # Generar la matriz de confusión
    matriz_confusion = confusion_matrix(y_test, y_pred)
    print("Matriz de confusión:")
    print(matriz_confusion)

    #Guardar modelo
    path_model = 'models/' + model_name + '_' + laboratoryID + '_' + spliteType + '.pkl'
    joblib.dump(model, path_model)

    return accuracy, precision, recall, matriz_confusion, path_model

def regresion_logreg1(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType):
    logreg1 = LogisticRegression(solver='liblinear', penalty='l2', C=1.0, max_iter=100)
    logreg1.fit(X_train, y_train)
    accuracy, precision, recall, matriz_confusion, path_model = evaluateAndSaveModel(X_test, y_test, logreg1, model_name, laboratoryID, spliteType)
    return accuracy, precision, recall, matriz_confusion, path_model

def regresion_logreg2(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType):
    logreg2 = LogisticRegression(solver='saga', penalty='l1', C=0.5, max_iter=1000)
    logreg2.fit(X_train, y_train)
    accuracy, precision, recall, matriz_confusion, path_model = evaluateAndSaveModel(X_test, y_test, logreg2, model_name, laboratoryID, spliteType)
    return accuracy, precision, recall, matriz_confusion, path_model

def regresion_logreg3(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logreg3 = LogisticRegression(solver='lbfgs', penalty='l2', C=0.1, multi_class='multinomial',max_iter=200)
    logreg3.fit(X_train_scaled, y_train)
    accuracy, precision, recall, matriz_confusion, path_model = evaluateAndSaveModel(X_test_scaled, y_test, logreg3, model_name, laboratoryID, spliteType)
    return accuracy, precision, recall, matriz_confusion, path_model

def SVM_linear(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType):
    svm_linear = SVC(kernel='linear', C=1.0, gamma='scale')
    svm_linear.fit(X_train, y_train)
    accuracy, precision, recall, matriz_confusion, path_model = evaluateAndSaveModel(X_test, y_test, svm_linear, model_name, laboratoryID, spliteType)
    return accuracy, precision, recall, matriz_confusion, path_model

def SVM_rbf(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm_rbf.fit(X_train_scaled, y_train)
    accuracy, precision, recall, matriz_confusion, path_model = evaluateAndSaveModel(X_test_scaled, y_test, svm_rbf, model_name, laboratoryID, spliteType)
    return accuracy, precision, recall, matriz_confusion, path_model

def SVM_poly(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svm_poly = SVC(kernel='poly', C=0.5, degree=3, gamma='auto')
    svm_poly.fit(X_train_scaled, y_train)
    accuracy, precision, recall, matriz_confusion, path_model = evaluateAndSaveModel(X_test_scaled, y_test, svm_poly, model_name, laboratoryID, spliteType)
    return accuracy, precision, recall, matriz_confusion, path_model

def decisionTrees_gini(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType):
    dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1)
    dt_gini.fit(X_train, y_train)
    accuracy, precision, recall, matriz_confusion, path_model = evaluateAndSaveModel(X_test, y_test, dt_gini, model_name, laboratoryID, spliteType)
    return accuracy, precision, recall, matriz_confusion, path_model

def decisionTrees_entropy(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType):
    dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=5, min_samples_leaf=2)
    dt_entropy.fit(X_train, y_train)
    accuracy, precision, recall, matriz_confusion, path_model = evaluateAndSaveModel(X_test, y_test, dt_entropy, model_name, laboratoryID,spliteType)
    return accuracy, precision, recall, matriz_confusion, path_model

def decisionTrees_randomized(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType):
    dt_randomized = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=4, min_samples_leaf=3, random_state=42)
    dt_randomized.fit(X_train, y_train)
    accuracy, precision, recall, matriz_confusion, path_model = evaluateAndSaveModel(X_test, y_test, dt_randomized, model_name, laboratoryID, spliteType)
    return accuracy, precision, recall, matriz_confusion, path_model

def randomForest_basic(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType):
    rf_basic = RandomForestClassifier(n_estimators=100, max_features='sqrt', max_depth=None, bootstrap=True)
    rf_basic.fit(X_train, y_train)
    accuracy, precision, recall, matriz_confusion, path_model = evaluateAndSaveModel(X_test, y_test, rf_basic, model_name, laboratoryID, spliteType)
    return accuracy, precision, recall, matriz_confusion, path_model




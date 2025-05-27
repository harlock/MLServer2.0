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
#from decouple import config
#from sklearn import *

def loadDT(algorithm, configuration, pathTraining, pathTest, laboratoryID, spliteType, target, features):
    # Cargar los conjuntos de datos divididos.
    X_train = pd.read_csv(pathTraining)
    X_test = pd.read_csv(pathTest)

    print(X_train)
    print(X_test)
    target = int(target)
    features = list(map(int, features.split(',')))
    print(target)
    print(features)

    y_train = X_train.iloc[:, target]    ##Target en train
    y_test = X_test.iloc[:, target]      ##Target en test

    X_train = X_train.iloc[:, features]  ##features en train
    X_test = X_test.iloc[:, features]    ##features en test

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

    precision = precision_score(y_test, y_pred, average='weighted') * 100
    precision = round(precision, 2)

    recall = recall_score(y_test, y_pred, average='weighted') * 100
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

def randomForest_large_trees(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType):
    rf_large_trees = RandomForestClassifier(n_estimators=300, max_features='sqrt', max_depth=10, bootstrap=True)
    rf_large_trees.fit(X_train, y_train)
    accuracy, precision, recall, matriz_confusion, path_model = evaluateAndSaveModel(X_test, y_test, rf_large_trees, model_name, laboratoryID, spliteType)
    return accuracy, precision, recall, matriz_confusion, path_model

def randomForest_regularized(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType):
    rf_regularized = RandomForestClassifier(n_estimators=500, max_features='log2', max_depth=15, min_samples_split=4, min_samples_leaf=2, bootstrap=True)
    rf_regularized.fit(X_train, y_train)
    accuracy, precision, recall, matriz_confusion, path_model = evaluateAndSaveModel(X_test, y_test, rf_regularized, model_name, laboratoryID, spliteType)
    return accuracy, precision, recall, matriz_confusion, path_model

def kNN_basic(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType):
    knn_basic = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
    knn_basic.fit(X_train, y_train)
    accuracy, precision, recall, matriz_confusion, path_model = evaluateAndSaveModel(X_test, y_test, knn_basic, model_name, laboratoryID, spliteType)
    return accuracy, precision, recall, matriz_confusion, path_model

#Probar especificamente este de abajo
def kNN_distance_weights(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    knn_distance_weights = KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='ball_tree')
    knn_distance_weights.fit(X_train_scaled, y_train)
    accuracy, precision, recall, matriz_confusion, path_model = evaluateAndSaveModel(X_test_scaled, y_test, knn_distance_weights, model_name, laboratoryID, spliteType)
    return accuracy, precision, recall, matriz_confusion, path_model

def kNN_manhattan(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    knn_manhattan = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='brute', metric='manhattan')
    knn_manhattan.fit(X_train_scaled, y_train)
    accuracy, precision, recall, matriz_confusion, path_model = evaluateAndSaveModel(X_test_scaled, y_test, knn_manhattan, model_name, laboratoryID, spliteType)
    return accuracy, precision, recall, matriz_confusion, path_model

def naiveBayes_gnb(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType):
    gnb = GaussianNB(var_smoothing=1e-9)
    gnb.fit(X_train, y_train)
    accuracy, precision, recall, matriz_confusion, path_model = evaluateAndSaveModel(X_test, y_test, gnb, model_name, laboratoryID, spliteType)
    return accuracy, precision, recall, matriz_confusion, path_model

def naiveBayes_mnb(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType):
    mnb = MultinomialNB(alpha=1.0, fit_prior=True)
    mnb.fit(X_train, y_train)
    accuracy, precision, recall, matriz_confusion, path_model = evaluateAndSaveModel(X_test, y_test, mnb, model_name, laboratoryID, spliteType)
    return accuracy, precision, recall, matriz_confusion, path_model

def naiveBayes_bnb(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType):
    binarizer = Binarizer(threshold=0.5)
    X_train_binarized = binarizer.fit_transform(X_train)
    X_test_binarized = binarizer.transform(X_test)
    bnb = BernoulliNB(alpha=0.5, binarize=0.0)
    bnb.fit(X_train_binarized, y_train)
    accuracy, precision, recall, matriz_confusion, path_model = evaluateAndSaveModel(X_test_binarized, y_test, bnb, model_name, laboratoryID, spliteType)
    return accuracy, precision, recall, matriz_confusion, path_model

def neural_basic(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType):
    nn_basic = MLPClassifier(hidden_layer_sizes=(100,), solver='adam', learning_rate='constant', max_iter=200)
    nn_basic.fit(X_train, y_train)
    accuracy, precision, recall, matriz_confusion, path_model = evaluateAndSaveModel(X_test, y_test, nn_basic, model_name, laboratoryID, spliteType)
    return accuracy, precision, recall, matriz_confusion, path_model

def neural_multi_layer(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType):
    nn_multi_layer = MLPClassifier(hidden_layer_sizes=(150, 100, 50), solver='adam', learning_rate='adaptive', max_iter=500)
    nn_multi_layer.fit(X_train, y_train)
    accuracy, precision, recall, matriz_confusion, path_model = evaluateAndSaveModel(X_test, y_test, nn_multi_layer, model_name, laboratoryID, spliteType)
    return accuracy, precision, recall, matriz_confusion, path_model

def neural_dropout_sgd(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType):
    nn_dropout_sgd = MLPClassifier(hidden_layer_sizes=(200, 100), solver='sgd', learning_rate='constant', max_iter=1000, alpha=0.001)
    nn_dropout_sgd.fit(X_train, y_train)
    accuracy, precision, recall, matriz_confusion, path_model = evaluateAndSaveModel(X_test, y_test, nn_dropout_sgd, model_name, laboratoryID, spliteType)
    return accuracy, precision, recall, matriz_confusion, path_model

def gradientBoosting_basic(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType):
    gb_basic = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0)
    gb_basic.fit(X_train, y_train)
    accuracy, precision, recall, matriz_confusion, path_model = evaluateAndSaveModel(X_test, y_test, gb_basic, model_name, laboratoryID, spliteType)
    return accuracy, precision, recall, matriz_confusion, path_model

def gradientBoosting_high_estimators(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType):
    gb_high_estimators = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8)
    gb_high_estimators.fit(X_train, y_train)
    accuracy, precision, recall, matriz_confusion, path_model = evaluateAndSaveModel(X_test, y_test, gb_high_estimators, model_name, laboratoryID, spliteType)
    return accuracy, precision, recall, matriz_confusion, path_model

def gradientBoosting_aggressive(X_train, y_train, X_test, y_test, model_name, laboratoryID, spliteType):
    gb_aggressive = GradientBoostingClassifier(n_estimators=500, learning_rate=0.01, max_depth=5, subsample=0.7)
    gb_aggressive.fit(X_train, y_train)
    accuracy, precision, recall, matriz_confusion, path_model = evaluateAndSaveModel(X_test, y_test, gb_aggressive, model_name, laboratoryID, spliteType)
    return accuracy, precision, recall, matriz_confusion, path_model

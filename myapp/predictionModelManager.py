import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR, LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score, precision_score, recall_score
from sklearn import svm

from sklearn import model_selection
from sklearn.metrics import classification_report



from celery import shared_task
from celery_progress.backend import ProgressRecorder
import time

from django.shortcuts import render

from .datasetManager import loadDataSetWithFeatures, loadDataSet

from .serviceManager import *

import json

from django.http import JsonResponse

from sklearn.ensemble import ExtraTreesClassifier

RANDOM_STATE = 42
FIG_SIZE = (10, 7)
TEST_SIZE = 0.30


class DatasetExplorer:
    
    features, target = None, None
    
    def __init__(self, *args, **kwargs):
        
        loadData = loadDataSet()
        self.target, self.features = loadData

    def mostImportantFeatures(self, n_features = 0):
        model = ExtraTreesClassifier()
        model.fit(self.features, self.target)
        feat_importances = pd.Series(model.feature_importances_, index=self.features.columns)
        if n_features == 0:
            n_features = len(self.features)
        
        most_important_feratures = feat_importances.nlargest(n_features)
        dictFeatures = dict(most_important_feratures)
        return dictFeatures


class ModelManager:
    
    
    features, target, jsonResponse = None, None, None
    
    estimator = SVC(kernel="linear")
    
    scaler = StandardScaler()
    
    testFeatures = None
    
    def __init__(self, testFeatures, *args, **kwargs):
        
        columns = list(testFeatures.keys())
        loadData = loadDataSetWithFeatures( selectedFeatures = columns )
        self.target, self.features = loadData
        self.testFeatures = testFeatures
    
    def datasetExists(self):
        return self.features != None


    def makeModel(self ):

        tAccScores = []
        
        t0AucScores = []
        t0PreScores = []
        t0RecScores = []
        
        t1AucScores = []
        t1PreScores = []
        t1RecScores = []
        
        trAccScores = []
        trAucScores = []
        trPreScores = []
        trRecScores = []
        
        predAlone = []
        
        predAll = []


        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target,
                                                    test_size=TEST_SIZE,
                                                    random_state=RANDOM_STATE)
                
        X_test = X_test.append(self.testFeatures, ignore_index=True)
        y_test_0 = y_test.append( pd.Series([0]) )
        y_test_1 = y_test.append( pd.Series([1]) )
        
        #--------------------------------- Reducing dimensions by PCA procedure ---------------------------------
        classifier = GaussianNB()
        components = len(list(self.testFeatures.keys()))
        
        std_clf = make_pipeline(StandardScaler(), PCA(n_components=components), classifier)
        std_clf.fit(X_train, y_train)
        
        pca_std = std_clf.named_steps['pca']
        scaler = std_clf.named_steps['standardscaler']
        X_train_std_transformed = pca_std.transform(scaler.transform(X_train))
        X_test_std_transformed = pca_std.transform(scaler.transform(X_test))
        print("--------------------------------- Reducing dimensions by PCA procedure ---------------------------------")
        #--------------------------------- Reducing dimensions by PCA procedure ---------------------------------
        # progress_recorder.set_progress(1, 8)
        
        #--------------------------------- Selecting the best features ---------------------------------
        #selectBestFeatures()
        #--------------------------------- Selecting the best features ---------------------------------
        
        
        #--------------------------------- Testing ---------------------------------
        #--------------------------------- LogisticRegression ---------------------------------
        lr = LogisticRegression(solver='lbfgs', C = 10, fit_intercept= False, class_weight='balanced' )
        lr.fit(X_train_std_transformed, y_train)
        pred_lr = lr.predict(self.testFeatures)
        pred_lr_all = lr.predict(X_test_std_transformed)
        tr_pred_lr = lr.predict(X_train_std_transformed)
        
        # progress_recorder.set_progress(2, 8)

        tAccScores.append( lr.score(self.testFeatures, [0] ) )
        
        t0AucScores.append( roc_auc_score(y_test_0, pred_lr_all) )
        t0PreScores.append( precision_score(y_test_0, pred_lr_all) )
        t0RecScores.append( recall_score(y_test_0, pred_lr_all) )
        
        t1AucScores.append( roc_auc_score(y_test_1, pred_lr_all) )
        t1PreScores.append( precision_score(y_test_1, pred_lr_all) )
        t1RecScores.append( recall_score(y_test_1, pred_lr_all) )
        
        trAccScores.append( lr.score(self.testFeatures, [1]) )
        trAucScores.append( roc_auc_score(y_train, tr_pred_lr) )
        trPreScores.append( precision_score(y_train, tr_pred_lr) )
        trRecScores.append( recall_score(y_train, tr_pred_lr) )
        
        predAlone.append(pred_lr[0])
        predAll.append(pred_lr_all[-1])
        
        # progress_recorder.set_progress(3, 8)
        print("--------------------------------- LogisticRegression ---------------------------------")
        #--------------------------------- SVC ---------------------------------
        svc = svm.SVC(gamma="scale", kernel='rbf', C=10)
        svc.fit(X_train_std_transformed, y_train)
        pred_svc = svc.predict(self.testFeatures)
        pred_svc_all = svc.predict(X_test_std_transformed)
        tr_pred_svc = svc.predict(X_train_std_transformed)
        
        # progress_recorder.set_progress(4, 8)

        tAccScores.append( svc.score(self.testFeatures, [0] ) )
        
        t0AucScores.append( roc_auc_score(y_test_0, pred_svc_all) )
        t0PreScores.append( precision_score(y_test_0, pred_svc_all) )
        t0RecScores.append( recall_score(y_test_0, pred_svc_all))
        
        t1AucScores.append( roc_auc_score(y_test_0, pred_svc_all) )
        t1PreScores.append( precision_score(y_test_0, pred_svc_all) )
        t1RecScores.append( recall_score(y_test_0, pred_svc_all))
        
        trAccScores.append( svc.score(self.testFeatures, [1]) )
        trAucScores.append( roc_auc_score(y_train, tr_pred_svc) )
        trPreScores.append( precision_score(y_train, tr_pred_svc) )
        trRecScores.append( recall_score(y_train, tr_pred_svc) )
        
        predAlone.append(pred_svc[0])
        predAll.append(pred_svc_all[-1])
        
        # progress_recorder.set_progress(5, 8)
        print("--------------------------------- SVC ---------------------------------")
        #--------------------------------- KNeighborsClassifier ---------------------------------
        neighborsClassifier = KNeighborsClassifier( n_neighbors=15, algorithm='brute', 
                                           weights='uniform', metric= 'chebyshev')
        neighborsClassifier.fit(X_train_std_transformed, y_train)
        pred_neighborsClassifier = neighborsClassifier.predict(self.testFeatures)
        pred_neighborsClassifier_all = neighborsClassifier.predict(X_test_std_transformed)
        tr_pred_neighborsClassifier = neighborsClassifier.predict(X_train_std_transformed)
        
        # progress_recorder.set_progress(6, 8)

        tAccScores.append( neighborsClassifier.score(self.testFeatures, [0] ) )
        
        t0AucScores.append( roc_auc_score(y_test_0, pred_neighborsClassifier_all) )
        t0PreScores.append( precision_score(y_test_0, pred_neighborsClassifier_all) )
        t0RecScores.append( recall_score(y_test_0, pred_neighborsClassifier_all))
        
        t1AucScores.append( roc_auc_score(y_test_0, pred_neighborsClassifier_all) )
        t1PreScores.append( precision_score(y_test_0, pred_neighborsClassifier_all) )
        t1RecScores.append( recall_score(y_test_0, pred_neighborsClassifier_all))
        
        trAccScores.append( neighborsClassifier.score(self.testFeatures, [1]) )
        trAucScores.append( roc_auc_score(y_train, tr_pred_neighborsClassifier) )
        trPreScores.append( precision_score(y_train, tr_pred_neighborsClassifier) )
        trRecScores.append( recall_score(y_train, tr_pred_neighborsClassifier) )
        
        predAlone.append(pred_neighborsClassifier[0])
        predAll.append(pred_neighborsClassifier_all[-1])

        # progress_recorder.set_progress(7, 8)
        print("--------------------------------- KNeighborsClassifier ---------------------------------")
        #--------------------------------- SVR ---------------------------------
        """
        svr = svm.SVR(gamma="scale", kernel='rbf', C=1)
        svr.fit(X_train_std_transformed, y_train)
        pred_svr = svr.predict(self.testFeatures)
        pred_svr_all = svr.predict(X_test_std_transformed)
        tr_pred_svr = svr.predict(X_train_std_transformed)
        
        tAccScores.append( svr.score(self.testFeatures, [0] ) )
        
        t0AucScores.append( roc_auc_score(y_test_0, pred_svr_all) )
        t0PreScores.append( precision_score(y_test_0, pred_svr_all) )
        t0RecScores.append( recall_score(y_test_0, pred_svr_all))
        
        t1AucScores.append( roc_auc_score(y_test_0, pred_svr_all) )
        t1PreScores.append( precision_score(y_test_0, pred_svr_all) )
        t1RecScores.append( recall_score(y_test_0, pred_svr_all))
        
        trAccScores.append( svr.score(self.testFeatures, [1]) )
        trAucScores.append( roc_auc_score(y_train, tr_pred_svr) )
        trPreScores.append( precision_score(y_train, tr_pred_svr) )
        trRecScores.append( recall_score(y_train, tr_pred_svr) )
        
        predAlone.append(pred_svr[0])
        predAll.append(pred_svr_all[-1])
        """
        #--------------------------------- Testing ---------------------------------
        
        columns = [ 'Test Accuracy', 'Test 0 AUC', 'Test 0 Precision', 'Test 0 Recall', 
                   'Test 1 AUC', 'Test 1 Precision', 'Test 1 Recall', 
                   'Train Accuracy', 'Train AUC', 'Train Precision', 'Train Recall', 'Pred_alone', 'Pred_all']
        estimators = ['LogisticRegression', 'SVC', 'KNeighborsClassifier']

        list_of_9ples = list(zip(tAccScores, t0AucScores, t0PreScores, t0RecScores,
                          t1AucScores, t1PreScores, t1RecScores, 
                                 trAccScores, trAucScores, trPreScores, trRecScores,
                                predAlone, predAll ))
        df = pd.DataFrame(list_of_9ples, index=estimators, columns = columns)

        # df.to_csv( "static/SummaryPrediction.csv", index=True)
        
        # progress_recorder.set_progress(8, 8)

        return selectBestEstimator(df)

    def logisticRegressionModel( self ):

        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target,
                                                    test_size=TEST_SIZE,
                                                    random_state=RANDOM_STATE)
        
        
        #--------------------------------- Reducing dimensions by PCA procedure ---------------------------------
        classifier = GaussianNB()
        components = len(list(self.testFeatures.keys()))
        
        std_clf = make_pipeline(StandardScaler(), PCA(n_components=components), classifier)
        std_clf.fit(X_train, y_train)
        
        pca_std = std_clf.named_steps['pca']
        scaler = std_clf.named_steps['standardscaler']
        X_train_std_transformed = pca_std.transform(scaler.transform(X_train))
        X_test_std_transformed = pca_std.transform(scaler.transform(X_test))
        print("--------------------------------- Reducing dimensions by PCA procedure ---------------------------------")
        #--------------------------------- Reducing dimensions by PCA procedure ---------------------------------
        
        #--------------------------------- Testing ---------------------------------
        #--------------------------------- LogisticRegression ---------------------------------
        lr = LogisticRegression(solver='lbfgs', C = 10, fit_intercept= False, class_weight='balanced' )
        lr.fit(X_train_std_transformed, y_train)
        # Log loss ---------------------------------
        seed = 7
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        scoring = 'neg_log_loss'
        results = model_selection.cross_val_score(lr, X_train_std_transformed, y_train, cv=kfold, scoring=scoring)
        print("Logloss: %.3f (%.3f)", (results.mean(), results.std()))
        neg_log_loss_mean = results.mean()
        neg_log_loss_std = results.std()
        scoring = 'neg_mean_squared_error'
        results = model_selection.cross_val_score(lr, X_train_std_transformed, y_train, cv=kfold, scoring=scoring)
        print("MSE: %.3f (%.3f)", (results.mean(), results.std()))
        neg_mean_squared_error_mean = results.mean()
        neg_mean_squared_error_std = results.std()
        scoring = 'r2'
        results = model_selection.cross_val_score(lr, X_train_std_transformed, y_train, cv=kfold, scoring=scoring)
        print("R^2: %.3f (%.3f)", (results.mean(), results.std()))
        r2_mean = results.mean()
        r2_std = results.std()
        predicted = lr.predict(X_test_std_transformed)
        regressionSummary = classification_report(y_test, predicted, output_dict = True)

        #--------------------------------- Testing ---------------------------------
        
        resultsDict = {"neg_log_loss": {"mean": neg_log_loss_mean, "std": neg_log_loss_std},
         "neg_mean_squared_error":{"mean": neg_mean_squared_error_mean, "std":neg_mean_squared_error_std},
         "r2": {"mean": r2_mean, "std": r2_std}, "regressionSummary":regressionSummary }


        return resultsDict


    def selectBestFeatures(self):
        
        #--------------------------------- Selecting the best features ---------------------------------
        selector = RFECV(self.estimator, step=1, cv=5)
        selector = selector.fit(X_train_std_transformed[:100], y_train[:100])
        
        selected_columns = self.features.columns[selector.support_ == True]
        X_selected_features = self.features[selected_columns]
        
        scaler.fit(X_selected_features)
        
        X_selected_features = scaler.transform(X_selected_features)
        #--------------------------------- Selecting the best features ---------------------------------


class ModelTestManager:

    def __init__(self, *args, **kwargs):
        print("Inside init method")

    @shared_task(bind = True)
    def makeModelTest(self, enteredFeatures):

        df = pd.DataFrame(enteredFeatures, index=[0])
        progress_recorder = ProgressRecorder(self)
        modelManager = ModelManager(df, selectedFeatures = enteredFeatures )
        modelManager.makeModel(progress_recorder)

        return selectBestEstimator()
        # return 'PREDICITON DONE'


def predictResults( request ):
    if request.method == 'POST':
        
        requestBodyJson = json.loads(request.body)
        serviceManager = ServiceManager( requestBodyJson )
        enteredFeatures = serviceManager.extractSelectedFeatures()

        if isinstance(enteredFeatures, ServiceResponse):
            return JsonResponse( enteredFeatures.json() )
        else:
            df = pd.DataFrame(enteredFeatures)
            try:
                modelManager = ModelManager(df, selectedFeatures = enteredFeatures)
                prediction, bestEstimator, bestMethodRecall = modelManager.makeModel()
                predictionString = "No"
                if prediction:
                    predictionString = "Yes"
                
                responseData = {'predictionString': predictionString, 'bestEstimator': bestEstimator, "bestMethodRecall": bestMethodRecall}
                response = responseJson(ResponseTypes.SUCCESS, responseData=responseData)
                return JsonResponse( response.json() )
            except InvalidFeatureNames:
                response = responseJson(ResponseTypes.INVALID_FEATURES_NAMES)
                return JsonResponse( response.json() )
            except Exception as inst:
                response = responseJson(ResponseTypes.OTHER, inst)
                return JsonResponse( response.json() )
    else:
        JsonResponse({"message": "The request must be POST" })

def logR_Prediction( request ):
    if request.method == 'POST':
        requestBodyJson = json.loads(request.body)
        serviceManager = ServiceManager( requestBodyJson )
        enteredFeatures = serviceManager.extractSelectedFeatures()
        if isinstance(enteredFeatures, ServiceResponse):
            return JsonResponse( enteredFeatures.json() )
        else:
            df = pd.DataFrame(enteredFeatures)
            try:
                modelManager = ModelManager(df, selectedFeatures = enteredFeatures)
                responseData = modelManager.logisticRegressionModel()
                response = responseJson(ResponseTypes.SUCCESS, responseData=responseData)
                return JsonResponse( response.json() )
            except InvalidFeatureNames:
                response = responseJson(ResponseTypes.INVALID_FEATURES_NAMES)
                return JsonResponse( response.json() )
            except Exception as inst:
                response = responseJson(ResponseTypes.OTHER, inst)
                return JsonResponse( response.json() )

    else:
        JsonResponse({"message": "The request must be POST method" })

def getFeatureScore( request ):
    print("getFeatureScore: " )
    if request.method == 'GET':
        try:
            datasetExplorer = DatasetExplorer()
            responseData = datasetExplorer.mostImportantFeatures()
            response = responseJson(ResponseTypes.SUCCESS, responseData=responseData)
            return JsonResponse( response.json() )
        except Exception as inst:
            response = responseJson(ResponseTypes.OTHER, inst)
            return JsonResponse( response.json() )

    else:
        JsonResponse({"message": "The request must be GET method" })

def selectBestEstimator(summary):

    # summary = pd.read_csv("static/SummaryPrediction.csv", index_col=0)
    bestEstimator = ''
    bestMethodRecall = 0
    for targ in [0,1]:
        
        tempBestEstimator = np.argmax(summary['Test '+str(targ)+' Recall'])
        tempRecallValue = summary['Test '+str(targ)+' Recall'][tempBestEstimator]
        
        if bestMethodRecall < tempRecallValue:
            bestMethodRecall = tempRecallValue
            bestEstimator = tempBestEstimator
    prediction = summary['Pred_all'][bestEstimator]

    return ( str(prediction), str(bestEstimator), str(bestMethodRecall) )
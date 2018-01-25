import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score,accuracy_score,fbeta_score,make_scorer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import collections
from Resampling import Resampling
from sklearn.metrics import make_scorer

my_beta = 3
ftwo_scorer = make_scorer(fbeta_score, beta=my_beta)

class MyRandomForestClassifier():
    'Implelments Random Forest Classifier'
    def randomForestNoOversampling(self,featureMatrix,phishingURLLabel,fakeFeatureMatrix,fakeLabels,technique,OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F2_Scores/'+OUTPUT_START+'-'+technique+'F2_RandomForestResultsNoOversampling.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            URL_Train = list(URL_Train)
            for everyFeature in fakeFeatureMatrix:
                URL_Train.append(everyFeature)
            
            Label_Train = list(Label_Train)
            for everyFakeLabel in fakeLabels:
                Label_Train.append(everyFakeLabel)
            parameters_RandomForest = {'n_estimators': [10, 100, 1000], 'criterion': ('gini', 'entropy'),
                                       'oob_score': (True, False), 'warm_start': (True, False)}
            estimator = RandomForestClassifier()
            clf = GridSearchCV(estimator, parameters_RandomForest, n_jobs=15,scoring=ftwo_scorer)
            clf.fit(URL_Train, Label_Train)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = fbeta_score(Label_Test, result,beta=my_beta, pos_label=1.0)
            predictionResult.write("\nThe f2_score is:" + str(f1Score))
            predictionResult.flush()
            # accuracy_matrix.append(f1Score)
            # accuracyScore = 0
            # aucMetric = 0
            # rocAucScore = 0
            # rocAucScoreMicro=0
            # rocAucScoreMacro=0
            # rocAucScoreWeighted=0
            # # accuracyScore = accuracy_score(Label_Test, result)
            # # aucMetric = auc(Label_Test, result, reorder=True)
            # rocAucScoreMicro = roc_auc_score(Label_Test, result,average='micro')
            # rocAucScoreMacro = roc_auc_score(Label_Test, result,average='macro')
            # rocAucScoreWeighted = roc_auc_score(Label_Test, result,average='weighted')

        except Exception as e:
            predictionResult.write(str(e))


        # predictionResult.write("\nROC_AUC_MICRO: " + str(rocAucScoreMicro))
        # predictionResult.write("\nROC_AUC_MACRO: " + str(rocAucScoreMacro))
        # predictionResult.write("\nROC_AUC_WEIGHTED: " + str(rocAucScoreWeighted))
        # predictionResult.write("\nAccuracy Score: " + str(accuracyScore))
        # predictionResult.write("\nAuc Metric: " + str(aucMetric))
        # predictionResult.write("\nRoc_Auc Score: " + str(rocAucScore))
        # predictionResult.write("Random Forest Classification without Oversampling Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))

    # ---- Random Forest Classifier with SMOTE ---------------------------------------

    def randomForestSMOTE(self,featureMatrix,phishingURLLabel,fakeFeatureMatrix,fakeLabels,technique,OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F2_Scores/'+OUTPUT_START+'-'+technique+'F2_RandomForestResultsSmote.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_RandomForest = {'n_estimators': [10, 100, 1000], 'criterion': ('gini', 'entropy'),
                                       'oob_score': (True, False), 'warm_start': (True, False)}

            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)
            estimator = RandomForestClassifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.smoteOversampling(URL_Train,Label_Train)
            clf = GridSearchCV(estimator, parameters_RandomForest, n_jobs=15,scoring=ftwo_scorer)
            clf.fit(featureMatrix2,phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = fbeta_score(Label_Test, result,beta=my_beta, pos_label=1.0)
            predictionResult.write("\nThe f2_score is:" + str(f1Score))
            predictionResult.flush()
            # accuracy_matrix.append(f1Score)
            # accuracyScore = 0
            # aucMetric = 0
            # rocAucScore = 0
            # rocAucScoreMicro=0
            # rocAucScoreMacro=0
            # rocAucScoreWeighted=0
            # # accuracyScore = accuracy_score(Label_Test, result)
            # # aucMetric = auc(Label_Test, result, reorder=True)
            # rocAucScoreMicro = roc_auc_score(Label_Test, result,average='micro')
            # rocAucScoreMacro = roc_auc_score(Label_Test, result,average='macro')
            # rocAucScoreWeighted = roc_auc_score(Label_Test, result,average='weighted')
        except Exception as e:
            predictionResult.write(str(e))

        # predictionResult.write("\nROC_AUC_MICRO: " + str(rocAucScoreMicro))
        # predictionResult.write("\nROC_AUC_MACRO: " + str(rocAucScoreMacro))
        # predictionResult.write("\nROC_AUC_WEIGHTED: " + str(rocAucScoreWeighted))

    # ------------- Random Forest Classifier with Borderline-1 Oversampling -------------------------------

    def randomForestb1SMOTE(self,featureMatrix,phishingURLLabel,fakeFeatureMatrix,fakeLabels,technique,OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F2_Scores/'+OUTPUT_START+'-'+technique+'F2_RandomForestResultsb1Smote.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_RandomForest = {'n_estimators': [10, 100, 1000], 'criterion': ('gini', 'entropy'),
                                       'oob_score': (True, False), 'warm_start': (True, False)}

            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)
            estimator = RandomForestClassifier()
            rm = Resampling()
            featureMatrix2,phishingLabel2 = rm.b1smoteOversampling(URL_Train,Label_Train)
            clf = GridSearchCV(estimator, parameters_RandomForest, n_jobs=15,scoring=ftwo_scorer)
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = fbeta_score(Label_Test, result,beta=my_beta, pos_label=1.0)
            predictionResult.write("\nThe f2_score is:" + str(f1Score))
            predictionResult.flush()
            # accuracy_matrix.append(f1Score)
            # accuracyScore = 0
            # aucMetric = 0
            # rocAucScore = 0
            # rocAucScoreMicro=0
            # rocAucScoreMacro=0
            # rocAucScoreWeighted=0
            # # accuracyScore = accuracy_score(Label_Test, result)
            # # aucMetric = auc(Label_Test, result, reorder=True)
            # rocAucScoreMicro = roc_auc_score(Label_Test, result,average='micro')
            # rocAucScoreMacro = roc_auc_score(Label_Test, result,average='macro')
            # rocAucScoreWeighted = roc_auc_score(Label_Test, result,average='weighted')
        except Exception as e:
            predictionResult.write(str(e))

        # predictionResult.write("\nROC_AUC_MICRO: " + str(rocAucScoreMicro))
        # predictionResult.write("\nROC_AUC_MACRO: " + str(rocAucScoreMacro))
        # predictionResult.write("\nROC_AUC_WEIGHTED: " + str(rocAucScoreWeighted))

    # ---------- Random Forest Classifier with Borderline-2 Oversampling -------------------------------------------------

    def randomForestb2SMOTE(self,featureMatrix,phishingURLLabel,fakeFeatureMatrix,fakeLabels,technique,OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F2_Scores/'+OUTPUT_START+'-'+technique+'F2_RandomForestResultsb2Smote.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_RandomForest = {'n_estimators': [10, 100, 1000], 'criterion': ('gini', 'entropy'),
                                       'oob_score': (True, False), 'warm_start': (True, False)}

            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)
            estimator = RandomForestClassifier()
            rm = Resampling()
            featureMatrix2,phishingLabel2 = rm.b2smoteOversampling(URL_Train,Label_Train)
            clf = GridSearchCV(estimator, parameters_RandomForest, n_jobs=15,scoring=ftwo_scorer)
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = fbeta_score(Label_Test, result,beta=my_beta, pos_label=1.0)
            predictionResult.write("\nThe f2_score is:" + str(f1Score))
            predictionResult.flush()
            # accuracy_matrix.append(f1Score)
            # accuracyScore = 0
            # aucMetric = 0
            # rocAucScore = 0
            # rocAucScoreMicro=0
            # rocAucScoreMacro=0
            # rocAucScoreWeighted=0
            # # accuracyScore = accuracy_score(Label_Test, result)
            # # aucMetric = auc(Label_Test, result, reorder=True)
            # rocAucScoreMicro = roc_auc_score(Label_Test, result,average='micro')
            # rocAucScoreMacro = roc_auc_score(Label_Test, result,average='macro')
            # rocAucScoreWeighted = roc_auc_score(Label_Test, result,average='weighted')
        except Exception as e:
            predictionResult.write(str(e))

        # predictionResult.write("\nROC_AUC_MICRO: " + str(rocAucScoreMicro))
        # predictionResult.write("\nROC_AUC_MACRO: " + str(rocAucScoreMacro))
        # predictionResult.write("\nROC_AUC_WEIGHTED: " + str(rocAucScoreWeighted))

    #----------------- Randome Forest Classifier with SVM SMOTE Oversampling ----------------------------------------------

    def randomForestSVM_Smote(self,featureMatrix,phishingURLLabel,fakeFeatureMatrix,fakeLabels,technique,OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F2_Scores/'+OUTPUT_START+'-'+technique+'F2_RandomForestResultsSVMSmote.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)

            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)
            parameters_RandomForest = {'n_estimators': [10, 100, 1000], 'criterion': ('gini', 'entropy'),
                                       'oob_score': (True, False), 'warm_start': (True, False)}
            estimator = RandomForestClassifier()
            rm = Resampling()
            featureMatrix2, phishingLablel2 = rm.SVMsmoteOversampling(URL_Train,Label_Train)
            clf = GridSearchCV(estimator, parameters_RandomForest, n_jobs=15,scoring=ftwo_scorer)
            clf.fit(featureMatrix2, phishingLablel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = fbeta_score(Label_Test, result,beta=my_beta, pos_label=1.0)
            predictionResult.write("\nThe f2_score is:" + str(f1Score))
            predictionResult.flush()
            # accuracy_matrix.append(f1Score)
            # accuracyScore = 0
            # aucMetric = 0
            # rocAucScore = 0
            # rocAucScoreMicro=0
            # rocAucScoreMacro=0
            # rocAucScoreWeighted=0
            # # accuracyScore = accuracy_score(Label_Test, result)
            # # aucMetric = auc(Label_Test, result, reorder=True)
            # rocAucScoreMicro = roc_auc_score(Label_Test, result,average='micro')
            # rocAucScoreMacro = roc_auc_score(Label_Test, result,average='macro')
            # rocAucScoreWeighted = roc_auc_score(Label_Test, result,average='weighted')
        except Exception as e:
            predictionResult.write(str(e))

        # predictionResult.write("\nROC_AUC_MICRO: " + str(rocAucScoreMicro))
        # predictionResult.write("\nROC_AUC_MACRO: " + str(rocAucScoreMacro))
        # predictionResult.write("\nROC_AUC_WEIGHTED: " + str(rocAucScoreWeighted))

    #--------------------------- Random Forest Classifier with Random Minority Oversampling with replacement ---------

    def randomForestRMR(self,featureMatrix,phishingURLLabel,fakeFeatureMatrix,fakeLabels,technique,OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F2_Scores/'+OUTPUT_START+'-'+technique+'F2_RandomForestResultsRMR.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)

            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)
            parameters_RandomForest = {'n_estimators': [10, 100, 1000], 'criterion': ('gini', 'entropy'),
                                       'oob_score': (True, False), 'warm_start': (True, False)}
            estimator = RandomForestClassifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.RMROversampling(URL_Train,Label_Train)
            clf = GridSearchCV(estimator, parameters_RandomForest, n_jobs=15,scoring=ftwo_scorer)
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = fbeta_score(Label_Test, result,beta=my_beta, pos_label=1.0)
            predictionResult.write("\nThe f2_score is:" + str(f1Score))
            predictionResult.flush()
            # accuracy_matrix.append(f1Score)
            # accuracyScore = 0
            # aucMetric = 0
            # rocAucScore = 0
            # rocAucScoreMicro=0
            # rocAucScoreMacro=0
            # rocAucScoreWeighted=0
            # # accuracyScore = accuracy_score(Label_Test, result)
            # # aucMetric = auc(Label_Test, result, reorder=True)
            # rocAucScoreMicro = roc_auc_score(Label_Test, result,average='micro')
            # rocAucScoreMacro = roc_auc_score(Label_Test, result,average='macro')
            # rocAucScoreWeighted = roc_auc_score(Label_Test, result,average='weighted')
        except Exception as e:
            predictionResult.write(str(e))

        # predictionResult.write("\nROC_AUC_MICRO: " + str(rocAucScoreMicro))
        # predictionResult.write("\nROC_AUC_MACRO: " + str(rocAucScoreMacro))
        # predictionResult.write("\nROC_AUC_WEIGHTED: " + str(rocAucScoreWeighted))
    #--------------------- Random Forest Classifier with ADASYN Oversampling -------------------------------------------

    def randomForestADASYN(self,featureMatrix,phishingURLLabel,fakeFeatureMatrix,fakeLabels,technique,OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F2_Scores/'+OUTPUT_START+'-'+technique+'F2_RandomForestResultsADASYN.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)

            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)
            parameters_RandomForest = {'n_estimators': [10, 100, 1000], 'criterion': ('gini', 'entropy'),
                                       'oob_score': (True, False), 'warm_start': (True, False)}
            estimator = RandomForestClassifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.ADASYNOversampling(URL_Train,Label_Train)
            clf = GridSearchCV(estimator, parameters_RandomForest, n_jobs=15,scoring=ftwo_scorer)
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = fbeta_score(Label_Test, result,beta=my_beta, pos_label=1.0)
            predictionResult.write("\nThe f2_score is:" + str(f1Score))
            predictionResult.flush()
            # accuracy_matrix.append(f1Score)
            # accuracyScore = 0
            # aucMetric = 0
            # rocAucScore = 0
            # rocAucScoreMicro=0
            # rocAucScoreMacro=0
            # rocAucScoreWeighted=0
            # # accuracyScore = accuracy_score(Label_Test, result)
            # # aucMetric = auc(Label_Test, result, reorder=True)
            # rocAucScoreMicro = roc_auc_score(Label_Test, result,average='micro')
            # rocAucScoreMacro = roc_auc_score(Label_Test, result,average='macro')
            # rocAucScoreWeighted = roc_auc_score(Label_Test, result,average='weighted')
        except Exception as e:
            predictionResult.write(str(e))

        # predictionResult.write("\nROC_AUC_MICRO: " + str(rocAucScoreMicro))
        # predictionResult.write("\nROC_AUC_MACRO: " + str(rocAucScoreMacro))
        # predictionResult.write("\nROC_AUC_WEIGHTED: " + str(rocAucScoreWeighted))

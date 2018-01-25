''' Implement KNN with F-1 scoring Classifier '''

import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
import collections
from Resampling import Resampling
from sklearn.neighbors import KNeighborsClassifier



class F1_KNN_Classifier():
    'Implements Ada boost classifier'

    # ---- Ada Boost Classifier without Oversampling
    def KNNNoOversampling(self, featureMatrix, phishingURLLabel, fakeFeatureMatrix, fakeLabels, technique,
                          OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F1_Scores/' + OUTPUT_START + '-' + technique + 'F1_KNNResultsNoOversampling.txt',
                                'a+')
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
            parameters_KNN = {'n_neighbors': [5, 10, 30,50], 'algorithm': ('auto', 'ball_tree','kd_tree')}
            estimator = KNeighborsClassifier()
            clf = GridSearchCV(estimator, parameters_KNN, n_jobs=15)
            clf.fit(URL_Train, Label_Train)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = f1_score(Label_Test, result,pos_label=1.0)
            predictionResult.write("\nThe f1_score is:" + str(f1Score))
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

    # -------------- Ada Boost Classifier with SMOTE Oversampling --------------------------------------
    def KNNBoostSMOTE(self, featureMatrix, phishingURLLabel, fakeFeatureMatrix, fakeLabels, technique, OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F1_Scores/' + OUTPUT_START + '-' + technique + 'F1_KNNResultsSmote.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_KNN = {'n_neighbors': [5, 10, 30, 50], 'algorithm': ('auto', 'ball_tree', 'kd_tree')}
            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)
            estimator = KNeighborsClassifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.smoteOversampling(URL_Train, Label_Train)
            clf = GridSearchCV(estimator, parameters_KNN, n_jobs=15)
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = f1_score(Label_Test, result, pos_label=1.0)
            predictionResult.write("\nThe f1_score is:" + str(f1Score))
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

    # --------- Ada Boost Classifier with Borderline-1 SMOTE----------------------------------------
    def KNNBoostb1SMOTE(self, featureMatrix, phishingURLLabel, fakeFeatureMatrix, fakeLabels, technique, OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F1_Scores/' + OUTPUT_START + '-' + technique + 'F1_KNNResultsSmoteb1.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_KNN = {'n_neighbors': [5, 10, 30, 50], 'algorithm': ('auto', 'ball_tree', 'kd_tree')}
            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)
            estimator = KNeighborsClassifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.b1smoteOversampling(URL_Train, Label_Train)
            clf = GridSearchCV(estimator, parameters_KNN, n_jobs=15)
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = f1_score(Label_Test, result, pos_label=1.0)
            predictionResult.write("\nThe f1_score is:" + str(f1Score))
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

    # ---------------- Ada Boost Classifier with Borderline-2 SMOTE ----------------------------

    def KNNb2SMOTE(self, featureMatrix, phishingURLLabel, fakeFeatureMatrix, fakeLabels, technique, OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F1_Scores/' + OUTPUT_START + '-' + technique + 'F1_KNNResultsSmoteb2.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_KNN = {'n_neighbors': [5, 10, 30, 50], 'algorithm': ('auto', 'ball_tree', 'kd_tree')}
            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)
            estimator = KNeighborsClassifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.b2smoteOversampling(URL_Train, Label_Train)
            clf = GridSearchCV(estimator, parameters_KNN, n_jobs=15)
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = f1_score(Label_Test, result,pos_label=1.0)
            predictionResult.write("\nThe f1_score is:" + str(f1Score))
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

    # ---------------------- Ada Boost Classifier with SVM Smote -----------------------------------------

    def KNNSVMSmote(self, featureMatrix, phishingURLLabel, fakeFeatureMatrix, fakeLabels, technique, OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F1_Scores/' + OUTPUT_START + '-' + technique + 'F1_KNNResultsSVMSmote.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_KNN = {'n_neighbors': [5, 10, 30, 50], 'algorithm': ('auto', 'ball_tree', 'kd_tree')}
            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)
            estimator = KNeighborsClassifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.SVMsmoteOversampling(URL_Train, Label_Train)
            clf = GridSearchCV(estimator, parameters_KNN, n_jobs=15)
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = f1_score(Label_Test, result, pos_label=1.0)
            predictionResult.write("\nThe f1_score is:" + str(f1Score))
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

    # --------------- Ada Boost Classifier with Random Minority Oversampling ------------------
    def KNNRMR(self, featureMatrix, phishingURLLabel, fakeFeatureMatrix, fakeLabels, technique, OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F1_Scores/' + OUTPUT_START + '-' + technique + 'F1_KNNResultsRMR.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_KNN = {'n_neighbors': [5, 10, 30, 50], 'algorithm': ('auto', 'ball_tree', 'kd_tree')}

            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)
            estimator = KNN_Classifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.RMROversampling(URL_Train, Label_Train)
            clf = GridSearchCV(estimator, parameters_KNN, n_jobs=15)
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = f1_score(Label_Test, result, pos_label=1.0)
            predictionResult.write("\nThe f1_score is:" + str(f1Score))
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

    # -------------------- ADa Boost Classifier with ADASYN Oversampling ---------------------------------------

    def KNNADASYN(self, featureMatrix, phishingURLLabel, fakeFeatureMatrix, fakeLabels, technique, OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F1_Scores/' + OUTPUT_START + '-' + technique + 'F1_KNNResultsADASYN.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_KNN = {'n_neighbors': [5, 10, 30, 50], 'algorithm': ('auto', 'ball_tree', 'kd_tree')}
            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)

            estimator = KNN_Classifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.ADASYNOversampling(URL_Train, Label_Train)
            clf = GridSearchCV(estimator, parameters_KNN, n_jobs=15)
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = f1_score(Label_Test, result,pos_label=1.0)
            predictionResult.write("\nThe f1_score is:" + str(f1Score))
            predictionResult.flush()
            # accuracy_matrix.append(f1Score)
            # accuracyScore = 0
            # aucMetric = 0
            # rocAucScore = 0
            # rocAucScoreMicro=0
            # rocAucScoreMacro=0
            # rocAucScoreWeighted=0
            # accuracyScore = accuracy_score(Label_Test, result)
            # aucMetric = auc(Label_Test, result, reorder=True)
            # rocAucScoreMicro = roc_auc_score(Label_Test, result,average='micro')
            # rocAucScoreMacro = roc_auc_score(Label_Test, result,average='macro')
            # rocAucScoreWeighted = roc_auc_score(Label_Test, result,average='weighted')
        except Exception as e:
            predictionResult.write(str(e))

            # predictionResult.write("\nROC_AUC_MICRO: " + str(rocAucScoreMicro))
            # predictionResult.write("\nROC_AUC_MACRO: " + str(rocAucScoreMacro))
            # predictionResult.write("\nROC_AUC_WEIGHTED: " + str(rocAucScoreWeighted))

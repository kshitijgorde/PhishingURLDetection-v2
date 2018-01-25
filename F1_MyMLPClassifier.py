''' Implement Multi-layer Perceptron with F-2 Scoring and changed later to Accuracy'''

import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score
import collections
from Resampling import Resampling
from sklearn.neural_network import MLPClassifier



class F1_MLP_Classifier():
    'Implements Ada boost classifier'

    # ---- Ada Boost Classifier without Oversampling
    def mlpNoOversampling(self, featureMatrix, phishingURLLabel, fakeFeatureMatrix, fakeLabels, technique,
                          OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(
            dir_name + '/F1_Scores/' + OUTPUT_START + '-' + technique + 'F1_MLPResultsNoOversampling.txt', 'a+')
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
            parameters_MLP = {'activation': ('identity', 'tanh', 'logistic', 'relu'),
                              'solver': ('lbfgs', 'sgd', 'adam'),
                              'learning_rate': ('constant', 'invscaling', 'adaptive'),
                              'max_iter': [200, 300, 400, 500]}
            estimator = MLPClassifier()
            clf = GridSearchCV(estimator, parameters_MLP, n_jobs=15,scoring='accuracy')
            clf.fit(URL_Train, Label_Train)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            #predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            # f1Score = f1_score(Label_Test, result, pos_label=1.0)
            # predictionResult.write("\nThe f1_score is:" + str(f1Score))
            predictionResult.flush()
            # accuracy_matrix.append(f1Score)
            # accuracyScore = 0
            # aucMetric = 0
            # rocAucScore = 0
            # rocAucScoreMicro=0
            # rocAucScoreMacro=0
            # rocAucScoreWeighted=0
            accuracyScore = accuracy_score(Label_Test, result)
            predictionResult.write("\nThe accuracy_score is:" + str(accuracyScore))
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
    def mlpBoostSMOTE(self, featureMatrix, phishingURLLabel, fakeFeatureMatrix, fakeLabels, technique, OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F1_Scores/' + OUTPUT_START + '-' + technique + 'F1_MLPResultsSmote.txt',
                                'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_MLP = {'activation': ('identity', 'tanh', 'logistic', 'relu'),
                              'solver': ('lbfgs', 'sgd', 'adam'),
                              'learning_rate': ('constant', 'invscaling', 'adaptive'),
                              'max_iter': [200, 300, 400, 500]}

            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)
            estimator = MLPClassifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.smoteOversampling(URL_Train, Label_Train)
            clf = GridSearchCV(estimator, parameters_MLP, n_jobs=15,scoring='accuracy')
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            #predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            # f1Score = f1_score(Label_Test, result, pos_label=1.0)
            # predictionResult.write("\nThe f1_score is:" + str(f1Score))
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
            accuracyScore = accuracy_score(Label_Test, result)
            predictionResult.write("\nThe accuracy_score is:" + str(accuracyScore))
        except Exception as e:
            predictionResult.write(str(e))

            # predictionResult.write("\nROC_AUC_MICRO: " + str(rocAucScoreMicro))
            # predictionResult.write("\nROC_AUC_MACRO: " + str(rocAucScoreMacro))
            # predictionResult.write("\nROC_AUC_WEIGHTED: " + str(rocAucScoreWeighted))

    # --------- Ada Boost Classifier with Borderline-1 SMOTE----------------------------------------
    def mlpBoostb1SMOTE(self, featureMatrix, phishingURLLabel, fakeFeatureMatrix, fakeLabels, technique, OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F1_Scores/' + OUTPUT_START + '-' + technique + 'F1_MLPResultsSmoteb1.txt',
                                'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)

            parameters_MLP = {'activation': ('identity', 'tanh', 'logistic', 'relu'),
                              'solver': ('lbfgs', 'sgd', 'adam'),
                              'learning_rate': ('constant', 'invscaling', 'adaptive'),
                              'max_iter': [200, 300, 400, 500]}
            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)
            estimator = MLPClassifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.b1smoteOversampling(URL_Train, Label_Train)
            clf = GridSearchCV(estimator, parameters_MLP, n_jobs=15,scoring='accuracy')
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            #predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            # f1Score = f1_score(Label_Test, result, pos_label=1.0)
            # predictionResult.write("\nThe f1_score is:" + str(f1Score))
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
            accuracyScore = accuracy_score(Label_Test, result)
            predictionResult.write("\nThe accuracy_score is:" + str(accuracyScore))
        except Exception as e:
            predictionResult.write(str(e))

            # predictionResult.write("\nROC_AUC_MICRO: " + str(rocAucScoreMicro))
            # predictionResult.write("\nROC_AUC_MACRO: " + str(rocAucScoreMacro))
            # predictionResult.write("\nROC_AUC_WEIGHTED: " + str(rocAucScoreWeighted))

    # ---------------- Ada Boost Classifier with Borderline-2 SMOTE ----------------------------

    def mlpb2SMOTE(self, featureMatrix, phishingURLLabel, fakeFeatureMatrix, fakeLabels, technique, OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F1_Scores/' + OUTPUT_START + '-' + technique + 'F1_MLPResultsSmoteb2.txt',
                                'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_MLP = {'activation': ('identity', 'tanh', 'logistic', 'relu'),
                              'solver': ('lbfgs', 'sgd', 'adam'),
                              'learning_rate': ('constant', 'invscaling', 'adaptive'),
                              'max_iter': [200, 300, 400, 500]}
            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)
            estimator = MLPClassifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.b2smoteOversampling(URL_Train, Label_Train)
            clf = GridSearchCV(estimator, parameters_MLP, n_jobs=15,scoring='accuracy')
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            #predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            # f1Score = f1_score(Label_Test, result,pos_label=1.0)
            # predictionResult.write("\nThe f1_score is:" + str(f1Score))
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
            accuracyScore = accuracy_score(Label_Test, result)
            predictionResult.write("\nThe accuracy_score is:" + str(accuracyScore))
        except Exception as e:
            predictionResult.write(str(e))

            # predictionResult.write("\nROC_AUC_MICRO: " + str(rocAucScoreMicro))
            # predictionResult.write("\nROC_AUC_MACRO: " + str(rocAucScoreMacro))
            # predictionResult.write("\nROC_AUC_WEIGHTED: " + str(rocAucScoreWeighted))

    # ---------------------- Ada Boost Classifier with SVM Smote -----------------------------------------

    def mlpSVMSmote(self, featureMatrix, phishingURLLabel, fakeFeatureMatrix, fakeLabels, technique, OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F1_Scores/' + OUTPUT_START + '-' + technique + 'F1_MLPResultsSVMSmote.txt',
                                'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)

            parameters_MLP = {'activation': ('identity', 'tanh', 'logistic', 'relu'),
                              'solver': ('lbfgs', 'sgd', 'adam'),
                              'learning_rate': ('constant', 'invscaling', 'adaptive'),
                              'max_iter': [200, 300, 400, 500]}
            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)
            estimator = MLPClassifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.SVMsmoteOversampling(URL_Train, Label_Train)
            clf = GridSearchCV(estimator, parameters_MLP, n_jobs=15,scoring='accuracy')
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            #predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            # f1Score = f1_score(Label_Test, result,pos_label=1.0)
            # predictionResult.write("\nThe f1_score is:" + str(f1Score))
            # predictionResult.flush()
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
            accuracyScore = accuracy_score(Label_Test, result)
            predictionResult.write("\nThe accuracy_score is:" + str(accuracyScore))
        except Exception as e:
            predictionResult.write(str(e))

            # predictionResult.write("\nROC_AUC_MICRO: " + str(rocAucScoreMicro))
            # predictionResult.write("\nROC_AUC_MACRO: " + str(rocAucScoreMacro))
            # predictionResult.write("\nROC_AUC_WEIGHTED: " + str(rocAucScoreWeighted))

    # --------------- Ada Boost Classifier with Random Minority Oversampling ------------------
    def mlpRMR(self, featureMatrix, phishingURLLabel, fakeFeatureMatrix, fakeLabels, technique, OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F1_Scores/' + OUTPUT_START + '-' + technique + 'F1_MLPResultsRMR.txt',
                                'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_MLP = {'activation': ('identity', 'tanh', 'logistic', 'relu'),
                              'solver': ('lbfgs', 'sgd', 'adam'),
                              'learning_rate': ('constant', 'invscaling', 'adaptive'),
                              'max_iter': [200, 300, 400, 500]}

            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)
            estimator = MLPClassifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.RMROversampling(URL_Train, Label_Train)
            clf = GridSearchCV(estimator, parameters_MLP, n_jobs=15,scoring='accuracy')
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            #predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            # f1Score = f1_score(Label_Test, result,pos_label=1.0)
            # predictionResult.write("\nThe f1_score is:" + str(f1Score))
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
            accuracyScore = accuracy_score(Label_Test, result)
            predictionResult.write("\nThe accuracy_score is:" + str(accuracyScore))
            # rocAucScoreWeighted = roc_auc_score(Label_Test, result,average='weighted')
        except Exception as e:
            predictionResult.write(str(e))

            # predictionResult.write("\nROC_AUC_MICRO: " + str(rocAucScoreMicro))
            # predictionResult.write("\nROC_AUC_MACRO: " + str(rocAucScoreMacro))
            # predictionResult.write("\nROC_AUC_WEIGHTED: " + str(rocAucScoreWeighted))

    # -------------------- ADa Boost Classifier with ADASYN Oversampling ---------------------------------------

    def mlpADASYN(self, featureMatrix, phishingURLLabel, fakeFeatureMatrix, fakeLabels, technique, OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F1_Scores/' + OUTPUT_START + '-' + technique + 'F1_MLPResultsADASYN.txt',
                                'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_MLP = {'activation': ('identity', 'tanh', 'logistic', 'relu'),
                              'solver': ('lbfgs', 'sgd', 'adam'),
                              'learning_rate': ('constant', 'invscaling', 'adaptive'),
                              'max_iter': [200, 300, 400, 500]}
            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)

            estimator = MLPClassifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.ADASYNOversampling(URL_Train, Label_Train)
            clf = GridSearchCV(estimator, parameters_MLP, n_jobs=15,scoring='accuracy')
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            #predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            # f1Score = f1_score(Label_Test, result,pos_label=1.0)
            # predictionResult.write("\nThe f1_score is:" + str(f1Score))
            # predictionResult.flush()
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
            accuracyScore = accuracy_score(Label_Test, result)
            predictionResult.write("\nThe accuracy_score is:" + str(accuracyScore))
        except Exception as e:
            predictionResult.write(str(e))

            # predictionResult.write("\nROC_AUC_MICRO: " + str(rocAucScoreMicro))
            # predictionResult.write("\nROC_AUC_MACRO: " + str(rocAucScoreMacro))
            # predictionResult.write("\nROC_AUC_WEIGHTED: " + str(rocAucScoreWeighted))

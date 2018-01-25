''' Implement Gaussian Process Classifier with F2 scoring '''

import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, fbeta_score, make_scorer
import collections
from Resampling import Resampling
from sklearn.gaussian_process import GaussianProcessClassifier

my_beta = 3
ftwo_scorer = make_scorer(fbeta_score, beta=my_beta)


class Gaussian_Classifier():
    'Implements Ada boost classifier'

    # ---- Ada Boost Classifier without Oversampling
    def gaussianNoOversampling(self, featureMatrix, phishingURLLabel, fakeFeatureMatrix, fakeLabels, technique,
                          OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F2_Scores/' + OUTPUT_START + '-' + technique + 'F2_GaussianResultsNoOversampling.txt',
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
            # parameters_adaBoost = {'n_estimators': [50, 100, 1000], 'algorithm': ('SAMME', 'SAMME.R')}
            # estimator = GaussianProcessClassifier()
            clf = GaussianProcessClassifier(n_jobs=15)
            clf.fit(URL_Train, Label_Train)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = fbeta_score(Label_Test, result, beta=my_beta, pos_label=1.0)
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

    # -------------- Ada Boost Classifier with SMOTE Oversampling --------------------------------------
    def gaussianBoostSMOTE(self, featureMatrix, phishingURLLabel, fakeFeatureMatrix, fakeLabels, technique, OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F2_Scores/' + OUTPUT_START + '-' + technique + 'F2_GaussianResultsSmote.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            #parameters_adaBoost = {'n_estimators': [50, 100, 1000], 'algorithm': ('SAMME', 'SAMME.R')}
            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)
            #estimator = GaussianProcessClassifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.smoteOversampling(URL_Train, Label_Train)
            clf = GaussianProcessClassifier(n_jobs=15)
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = fbeta_score(Label_Test, result, beta=my_beta, pos_label=1.0)
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

    # --------- Ada Boost Classifier with Borderline-1 SMOTE----------------------------------------
    def gaussianBoostb1SMOTE(self, featureMatrix, phishingURLLabel, fakeFeatureMatrix, fakeLabels, technique, OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F2_Scores/' + OUTPUT_START + '-' + technique + 'F2_GaussianResultsSmoteb1.txt', 'a+')
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
            #estimator = AdaBoostClassifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.b1smoteOversampling(URL_Train, Label_Train)
            clf = GaussianProcessClassifier(n_jobs=15)
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = fbeta_score(Label_Test, result, beta=my_beta, pos_label=1.0)
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

    # ---------------- Ada Boost Classifier with Borderline-2 SMOTE ----------------------------

    def gaussianb2SMOTE(self, featureMatrix, phishingURLLabel, fakeFeatureMatrix, fakeLabels, technique, OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F2_Scores/' + OUTPUT_START + '-' + technique + 'F2_GaussianResultsSmoteb2.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_adaBoost = {'n_estimators': [50, 100, 1000], 'algorithm': ('SAMME', 'SAMME.R')}
            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)
            #estimator = AdaBoostClassifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.b2smoteOversampling(URL_Train, Label_Train)
            clf = GaussianProcessClassifier(n_jobs=15)
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = fbeta_score(Label_Test, result, beta=my_beta, pos_label=1.0)
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

    # ---------------------- Ada Boost Classifier with SVM Smote -----------------------------------------

    def gaussianSVMSmote(self, featureMatrix, phishingURLLabel, fakeFeatureMatrix, fakeLabels, technique, OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F2_Scores/' + OUTPUT_START + '-' + technique + 'F2_GaussianResultsSVMSmote.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_adaBoost = {'n_estimators': [50, 100, 1000], 'algorithm': ('SAMME', 'SAMME.R')}
            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)
            #estimator = AdaBoostClassifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.SVMsmoteOversampling(URL_Train, Label_Train)
            clf = GaussianProcessClassifier(n_jobs=15)
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = fbeta_score(Label_Test, result, beta=my_beta, pos_label=1.0)
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

    # --------------- Ada Boost Classifier with Random Minority Oversampling ------------------
    def gaussianRMR(self, featureMatrix, phishingURLLabel, fakeFeatureMatrix, fakeLabels, technique, OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F2_Scores/' + OUTPUT_START + '-' + technique + 'F2_GaussianResultsRMR.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_adaBoost = {'n_estimators': [50, 100, 1000], 'algorithm': ('SAMME', 'SAMME.R')}

            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)
            #estimator = AdaBoostClassifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.RMROversampling(URL_Train, Label_Train)
            clf = GaussianProcessClassifier(n_jobs=15)
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = fbeta_score(Label_Test, result, beta=my_beta, pos_label=1.0)
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

    # -------------------- ADa Boost Classifier with ADASYN Oversampling ---------------------------------------

    def gaussianADASYN(self, featureMatrix, phishingURLLabel, fakeFeatureMatrix, fakeLabels, technique, OUTPUT_START):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/F2_Scores/' + OUTPUT_START + '-' + technique + 'F2_GaussianResultsADASYN.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_adaBoost = {'n_estimators': [50, 100, 1000], 'algorithm': ('SAMME', 'SAMME.R')}
            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)

            #estimator = AdaBoostClassifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.ADASYNOversampling(URL_Train, Label_Train)
            clf = GaussianProcessClassifier(n_jobs=15)
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = fbeta_score(Label_Test, result, beta=my_beta, pos_label=1.0)
            predictionResult.write("\nThe f2_score is:" + str(f1Score))
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

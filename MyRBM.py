from sklearn.neural_network import BernoulliRBM
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score,accuracy_score
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import os
from sklearn.svm import SVC
import collections
from Resampling import Resampling
class MyRBM():
    'implements Bernoulli Restricted Boltzmann Classifier'
    def rbmClassifyNoOversampling(self,featureMatrix, phishingURLLabel):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/RBMResultsNoOversampling.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            estimator = BernoulliRBM()
            svm = SVC()
            clf = Pipeline(steps=[('rbm', estimator), ('SVC', svm)])
            clf.fit(URL_Train, Label_Train)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = f1_score(Label_Test, result, pos_label='1', average='macro')
            predictionResult.write("\nThe f1_score is:" + str(f1Score))
            predictionResult.flush()
            accuracy_matrix.append(f1Score)
        except Exception as e:
            predictionResult.write(str(e))

        predictionResult.write("RBM Classification without Oversampling Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))

    #---------- RBM Classifier with SMOTE Oversampling-----------------------------------------------------

    def rbmClassifySMOTE(self,featureMatrix, phishingURLLabel):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/RBMResultsSMOTE.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            estimator = BernoulliRBM()
            svm = SVC()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.ADASYNOversampling(URL_Train, Label_Train)
            clf = Pipeline(steps=[('rbm', estimator), ('SVC', svm)])
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = f1_score(Label_Test, result, pos_label='1', average='macro')
            predictionResult.write("\nThe f1_score is:" + str(f1Score))
            predictionResult.flush()
            accuracy_matrix.append(f1Score)
        except Exception as e:
            predictionResult.write(str(e))

        predictionResult.write("RBM Classification with SMOTE Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))

    # ------------------- RBM with borderline-1 SMOTE
    def rbmClassifyb1Smote(self,featureMatrix, phishingURLLabel):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/RBMResultsb1Smote.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            estimator = BernoulliRBM()
            svm = SVC()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.ADASYNOversampling(URL_Train, Label_Train)
            clf = Pipeline(steps=[('rbm', estimator), ('SVC', svm)])
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = f1_score(Label_Test, result, pos_label='1', average='macro')
            predictionResult.write("\nThe f1_score is:" + str(f1Score))
            predictionResult.flush()
            accuracy_matrix.append(f1Score)
        except Exception as e:
            predictionResult.write(str(e))

        predictionResult.write("RBM Classification with Borderline-1 Smote Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))

    #--------------- RBM with borderline-2 smote

    def rbmClassifyb2Smote(self,featureMatrix, phishingURLLabel):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/RBMResultsb2Smote.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            estimator = BernoulliRBM()
            svm = SVC()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.ADASYNOversampling(URL_Train, Label_Train)
            clf = Pipeline(steps=[('rbm', estimator), ('SVC', svm)])
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = f1_score(Label_Test, result, pos_label='1', average='macro')
            predictionResult.write("\nThe f1_score is:" + str(f1Score))
            predictionResult.flush()
            accuracy_matrix.append(f1Score)
        except Exception as e:
            predictionResult.write(str(e))

        predictionResult.write("RBM Classification with Borderline-2 Smote Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))

    # ------------------- RBM with SVM SMote-------------------------------------

    def rbmClassifySVMSmote(self,featureMatrix, phishingURLLabel):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/RBMResultsSVMSmote.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            estimator = BernoulliRBM()
            svm = SVC()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.ADASYNOversampling(URL_Train, Label_Train)
            clf = Pipeline(steps=[('rbm', estimator), ('SVC', svm)])
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = f1_score(Label_Test, result, pos_label='1', average='macro')
            predictionResult.write("\nThe f1_score is:" + str(f1Score))
            predictionResult.flush()
            accuracy_matrix.append(f1Score)
        except Exception as e:
            predictionResult.write(str(e))

        predictionResult.write("RBM Classification with SVM Smote Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))

    #------------------ RBM with RMR------------------------------
    def rbmClassifyRMR(self,featureMatrix, phishingURLLabel):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/RBMResultsRMR.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            estimator = BernoulliRBM()
            svm = SVC()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.ADASYNOversampling(URL_Train, Label_Train)
            clf = Pipeline(steps=[('rbm', estimator), ('SVC', svm)])
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = f1_score(Label_Test, result, pos_label='1', average='macro')
            predictionResult.write("\nThe f1_score is:" + str(f1Score))
            predictionResult.flush()
            accuracy_matrix.append(f1Score)
        except Exception as e:
            predictionResult.write(str(e))

        predictionResult.write("RBM Classification with RMR Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))

    #---------------------- RBM with ADASYN -----------------------------------------------------------------------

    def rbmClassifyADASYN(self,featureMatrix, phishingURLLabel):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/RBMResultsAdaSyn.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            estimator = BernoulliRBM()
            svm = SVC()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.ADASYNOversampling(URL_Train, Label_Train)
            clf = Pipeline(steps=[('rbm', estimator), ('SVC', svm)])
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = f1_score(Label_Test, result, pos_label='1', average='macro')
            predictionResult.write("\nThe f1_score is:" + str(f1Score))
            predictionResult.flush()
            accuracy_matrix.append(f1Score)
        except Exception as e:
            predictionResult.write(str(e))

        predictionResult.write("RBM Classification with ADASYN Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))
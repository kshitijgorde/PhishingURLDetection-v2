from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score,accuracy_score,make_scorer,fbeta_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from Resampling import Resampling
import numpy

import os
import collections
from sklearn.metrics import auc,roc_auc_score

my_beta = 3
ftwo_scorer = make_scorer(fbeta_score, beta=my_beta)
class MyDecisionTreeClassifier():
    'Handles Predicting Phishing URL by implementing scikit-learn DecisionTree Classifier'

    def decisionTreeSMOTE(self,featureMatrix,phishingURLLabel,fakeFeatureMatrix,fakeLabels,technique,OUTPUT_START):
        re = Resampling()
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name+'/'+OUTPUT_START+'-'+technique+'DecisionTreeResultsSmote.txt','a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix,phishingURLLabel,test_size=0.20,random_state=40)

            #----------- Analysis
            URL_Train = list(URL_Train)
            for everyFeature in fakeFeatureMatrix:
                URL_Train.append(everyFeature)
            
            Label_Train = list(Label_Train)
            for everyFakeLabel in fakeLabels:
                Label_Train.append(everyFakeLabel)

            print 'Train Test Split:'
            print 'Training Values:'
            print 'Total:' + str(len(Label_Train))
            print 'Phishy: '+str(list(Label_Train).count(1))
            print 'Non Phishy:' + str(list(Label_Train).count(0))

            print 'Testing Values:'
            print 'Total:' + str(len(Label_Test))
            print 'Phishy: ' + str(list(Label_Test).count(1))
            print 'Non Phishy:' + str(list(Label_Test).count(0))
            print 'Performing Oversampling'
            featureMatrix2, phishingLabel2 = re.smoteOversampling(URL_Train, Label_Train)

            print 'After Oversampling...'
            print 'Total: '+str(len(phishingLabel2))
            print 'Ratio: '
            print collections.Counter(phishingLabel2)

            parameters_DecisionTree = {'criterion': ('gini', 'entropy'), 'splitter': ('best', 'random')}
            estimator = DecisionTreeClassifier()
            # totalSamples = len(Label_Train)
            # positiveCount = int(Label_Train.count('1'))       #should be 65% of total
            # predictionResult.write("Percentage of positive samples in training phase: %.2f " % (positiveCount/float(totalSamples)))
            clf = GridSearchCV(estimator, parameters_DecisionTree, n_jobs=15,scoring=ftwo_scorer)
            URL_Test = list(URL_Test)
            featureMatrix2 = list(featureMatrix2)
            phishingLabel2 = list(phishingLabel2)
            clf.fit(featureMatrix2,phishingLabel2)
            result = clf.predict(URL_Test)
            print "Type of REsult is:"
            print type(result)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = fbeta_score(Label_Test, result,beta = my_beta,pos_label=1.0)
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

    #------------- Decision Tree Classifier with Random Minority Over-sampling with Replacement

    def decisionTreeRMR(self,featureMatrix,phishingURLLabel,fakeFeatureMatrix,fakeLabels,technique,OUTPUT_START):
        re = Resampling()
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name+'/'+OUTPUT_START+'-'+technique+'DecisionTreeResultsRMR.txt','a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix,phishingURLLabel,test_size=0.20,random_state=40)

            #----------- Analysis
            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)

            print 'Performing Oversampling'
            featureMatrix2, phishingLabel2 = re.RMROversampling(URL_Train, Label_Train)

            print 'After Oversampling...'
            print 'Total: '+str(len(phishingLabel2))
            print 'Ratio: '
            print collections.Counter(phishingLabel2)

            parameters_DecisionTree = {'criterion': ('gini', 'entropy'), 'splitter': ('best', 'random')}
            estimator = DecisionTreeClassifier()
            # totalSamples = len(Label_Train)
            # positiveCount = int(Label_Train.count('1'))       #should be 65% of total
            # predictionResult.write("Percentage of positive samples in training phase: %.2f " % (positiveCount/float(totalSamples)))
            clf = GridSearchCV(estimator, parameters_DecisionTree, n_jobs=15,scoring=ftwo_scorer)
            URL_Test = list(URL_Test)
            featureMatrix2 = list(featureMatrix2)
            phishingLabel2 = list(phishingLabel2)
            clf.fit(featureMatrix2,phishingLabel2)
            result = clf.predict(URL_Test)
            print "Type of REsult is:"
            print type(result)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = fbeta_score(Label_Test, result,beta = my_beta,pos_label=1.0)
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

    #---------- Decision Tree Classifier with Borderline SMOTE-1 ----------------------------

    def decisionTreebSMOTE1(self,featureMatrix,phishingURLLabel,fakeFeatureMatrix,fakeLabels,technique,OUTPUT_START):
        re = Resampling()
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name+'/'+OUTPUT_START+'-'+technique+'DecisionTreeResultsSmoteB1.txt','a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix,phishingURLLabel,test_size=0.20,random_state=40)

            #----------- Analysis
            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)

            print 'Train Test Split:'
            print 'Training Values:'
            print 'Total:' + str(len(Label_Train))
            print 'Phishy: '+str(list(Label_Train).count(1))
            print 'Non Phishy:' + str(list(Label_Train).count(0))

            print 'Testing Values:'
            print 'Total:' + str(len(Label_Test))
            print 'Phishy: ' + str(list(Label_Test).count(1))
            print 'Non Phishy:' + str(list(Label_Test).count(0))
            print 'Performing Oversampling'
            featureMatrix2, phishingLabel2 = re.b1smoteOversampling(URL_Train, Label_Train)

            print 'After Oversampling...'
            print 'Total: '+str(len(phishingLabel2))
            print 'Ratio: '
            print collections.Counter(phishingLabel2)

            parameters_DecisionTree = {'criterion': ('gini', 'entropy'), 'splitter': ('best', 'random')}
            estimator = DecisionTreeClassifier()
            # totalSamples = len(Label_Train)
            # positiveCount = int(Label_Train.count('1'))       #should be 65% of total
            # predictionResult.write("Percentage of positive samples in training phase: %.2f " % (positiveCount/float(totalSamples)))
            clf = GridSearchCV(estimator, parameters_DecisionTree, n_jobs=15,scoring=ftwo_scorer)
            URL_Test = list(URL_Test)
            featureMatrix2 = list(featureMatrix2)
            phishingLabel2 = list(phishingLabel2)
            clf.fit(featureMatrix2,phishingLabel2)
            result = clf.predict(URL_Test)
            #print "Type of REsult is:"
            #print type(result)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = fbeta_score(Label_Test, result,beta = my_beta,pos_label=1.0)
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

    # ------------------------- Decision Tree Classifier with Borderline SMOTE 2 ---------------------------

    def decisionTreebSMOTE2(self,featureMatrix,phishingURLLabel,fakeFeatureMatrix,fakeLabels,technique,OUTPUT_START):
        re = Resampling()
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name+'/'+OUTPUT_START+'-'+technique+'DecisionTreeResultsSmoteb2.txt','a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix,phishingURLLabel,test_size=0.20,random_state=40)

            #----------- Analysis
            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)
            print 'Train Test Split:'
            print 'Training Values:'
            print 'Total:' + str(len(Label_Train))
            print 'Phishy: '+str(list(Label_Train).count(1))
            print 'Non Phishy:' + str(list(Label_Train).count(0))

            print 'Testing Values:'
            print 'Total:' + str(len(Label_Test))
            print 'Phishy: ' + str(list(Label_Test).count(1))
            print 'Non Phishy:' + str(list(Label_Test).count(0))
            print 'Performing Oversampling'
            featureMatrix2, phishingLabel2 = re.b2smoteOversampling(URL_Train, Label_Train)

            print 'After Oversampling...'
            print 'Total: '+str(len(phishingLabel2))
            print 'Ratio: '
            print collections.Counter(phishingLabel2)

            parameters_DecisionTree = {'criterion': ('gini', 'entropy'), 'splitter': ('best', 'random')}
            estimator = DecisionTreeClassifier()
            # totalSamples = len(Label_Train)
            # positiveCount = int(Label_Train.count('1'))       #should be 65% of total
            # predictionResult.write("Percentage of positive samples in training phase: %.2f " % (positiveCount/float(totalSamples)))
            clf = GridSearchCV(estimator, parameters_DecisionTree, n_jobs=15,scoring=ftwo_scorer)
            URL_Test = list(URL_Test)
            featureMatrix2 = list(featureMatrix2)
            phishingLabel2 = list(phishingLabel2)
            clf.fit(featureMatrix2,phishingLabel2)
            result = clf.predict(URL_Test)
            #print "Type of REsult is:"
            #print type(result)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = fbeta_score(Label_Test, result,beta = my_beta,pos_label=1.0)
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

    # Decision Tree Classifier for Support Vector Machine SMOTE Technique

    def decisionTreeSVM_SMOTE(self,featureMatrix,phishingURLLabel,fakeFeatureMatrix,fakeLabels,technique,OUTPUT_START):
        re = Resampling()
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name+'/'+OUTPUT_START+'-'+technique+'DecisionTreeResultsSVMSmote.txt','a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix,phishingURLLabel,test_size=0.20,random_state=40)

            #----------- Analysis
            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)
            print 'Train Test Split:'
            print 'Training Values:'
            print 'Total:' + str(len(Label_Train))
            print 'Phishy: '+str(list(Label_Train).count(1))
            print 'Non Phishy:' + str(list(Label_Train).count(0))

            print 'Testing Values:'
            print 'Total:' + str(len(Label_Test))
            print 'Phishy: ' + str(list(Label_Test).count(1))
            print 'Non Phishy:' + str(list(Label_Test).count(0))
            print 'Performing Oversampling'
            featureMatrix2, phishingLabel2 = re.SVMsmoteOversampling(URL_Train, Label_Train)

            print 'After Oversampling...'
            print 'Total: '+str(len(phishingLabel2))
            print 'Ratio: '
            print collections.Counter(phishingLabel2)

            parameters_DecisionTree = {'criterion': ('gini', 'entropy'), 'splitter': ('best', 'random')}
            estimator = DecisionTreeClassifier()
            # totalSamples = len(Label_Train)
            # positiveCount = int(Label_Train.count('1'))       #should be 65% of total
            # predictionResult.write("Percentage of positive samples in training phase: %.2f " % (positiveCount/float(totalSamples)))
            clf = GridSearchCV(estimator, parameters_DecisionTree, n_jobs=15,scoring=ftwo_scorer)
            URL_Test = list(URL_Test)
            featureMatrix2 = list(featureMatrix2)
            phishingLabel2 = list(phishingLabel2)
            clf.fit(featureMatrix2,phishingLabel2)
            result = clf.predict(URL_Test)
            #print "Type of REsult is:"
            #print type(result)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = fbeta_score(Label_Test, result,beta = my_beta,pos_label=1.0)
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
    # --------------- Decision Tree Classifier with ADASYN (Adaptive Synthetic Sampling Approach for imbalanced Learning---

    def decisionTreeADASYN(self,featureMatrix,phishingURLLabel,fakeFeatureMatrix,fakeLabels,technique,OUTPUT_START):
        re = Resampling()
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name+'/'+OUTPUT_START+'-'+technique+'DecisionTreeResultsADASYN.txt','a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix,phishingURLLabel,test_size=0.20,random_state=40)

            #----------- Analysis
            URL_Train = list(URL_Train)
            # for everyFeature in fakeFeatureMatrix:
            #     URL_Train.append(everyFeature)
            #
            # Label_Train = list(Label_Train)
            # for everyFakeLabel in fakeLabels:
            #     Label_Train.append(everyFakeLabel)

            print 'Train Test Split:'
            print 'Training Values:'
            print 'Total:' + str(len(Label_Train))
            print 'Phishy: '+str(list(Label_Train).count(1))
            print 'Non Phishy:' + str(list(Label_Train).count(0))

            print 'Testing Values:'
            print 'Total:' + str(len(Label_Test))
            print 'Phishy: ' + str(list(Label_Test).count(1))
            print 'Non Phishy:' + str(list(Label_Test).count(0))
            print 'Performing Oversampling'
            featureMatrix2, phishingLabel2 = re.ADASYNOversampling(URL_Train, Label_Train)

            print 'After Oversampling...'
            print 'Total: '+str(len(phishingLabel2))
            print 'Ratio: '
            print collections.Counter(phishingLabel2)

            parameters_DecisionTree = {'criterion': ('gini', 'entropy'), 'splitter': ('best', 'random')}
            estimator = DecisionTreeClassifier()
            # totalSamples = len(Label_Train)
            # positiveCount = int(Label_Train.count('1'))       #should be 65% of total
            # predictionResult.write("Percentage of positive samples in training phase: %.2f " % (positiveCount/float(totalSamples)))
            clf = GridSearchCV(estimator, parameters_DecisionTree, n_jobs=15,scoring=ftwo_scorer)
            URL_Test = list(URL_Test)
            featureMatrix2 = list(featureMatrix2)
            phishingLabel2 = list(phishingLabel2)
            clf.fit(featureMatrix2,phishingLabel2)
            result = clf.predict(URL_Test)
            #print "Type of REsult is:"
            #print type(result)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = fbeta_score(Label_Test, result,beta = my_beta,pos_label=1.0)
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

    # Decision Tree Classifier without any Oversampling Technique

    def decisionTreeNoOversampling(self,featureMatrix,phishingURLLabel,fakeFeatureMatrix,fakeLabels,technique,OUTPUT_START):
        re = Resampling()
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name+'/'+OUTPUT_START+'-'+technique+'DecisionTreeResultsNoOversampling.txt','a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
	        URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix,phishingURLLabel,test_size=0.20)
	        print 'Originally length of URL_Train is:'
	        print len(URL_Train)
	        #----------- Analysis

	        #URL_Train = list(URL_Train)
	        # for everyFeature in fakeFeatureMatrix:
	        #     URL_Train.append(everyFeature)
	        #
	        # Label_Train = list(Label_Train)
	        # for everyFakeLabel in fakeLabels:
	        #     Label_Train.append(everyFakeLabel)

	        # URL_Test = list(URL_Test)
	        # URL_Train = list(URL_Train)
	        # Label_Train = list(Label_Train)
	        print URL_Test.shape
	        print URL_Train.shape
	        print Label_Train.shape
	        parameters_DecisionTree = {'criterion': ('gini', 'entropy'), 'splitter': ('best', 'random')}
	        estimator = DecisionTreeClassifier()
	        clf = GridSearchCV(estimator, parameters_DecisionTree, n_jobs=15,scoring=ftwo_scorer)


	        clf.fit(URL_Train,Label_Train)
	        result = clf.predict(URL_Test)
	        predictionResult.write(str(result))
	        predictionResult.flush()
	        # predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
	        predictionResult.flush()
	        f1Score = fbeta_score(Label_Test, result,beta = my_beta,pos_label=1.0)
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
	        # # aucMetric = auc(Label_Test, result,reorder=True)
	        # rocAucScoreMicro = roc_auc_score(Label_Test, result,average='micro')
	        # rocAucScoreMacro = roc_auc_score(Label_Test, result,average='macro')
	        # rocAucScoreWeighted = roc_auc_score(Label_Test, result,average='weighted')
        except Exception as e:
            predictionResult.write(str(e))

        # predictionResult.write("\nROC_AUC_MICRO: " + str(rocAucScoreMicro))
        # predictionResult.write("\nROC_AUC_MACRO: " + str(rocAucScoreMacro))
        # predictionResult.write("\nROC_AUC_WEIGHTED: " + str(rocAucScoreWeighted))

from LoadCSVDataset import LoadCSVDataset
from FeatureCSV import FeaturesCSV
from LoadFeatures import LoadFeatures
from GanPreProcess import GanPreProcess
import numpy
from F2_DecisionTreeClassifier import MyDecisionTreeClassifier
from F2_MyRandomForest import MyRandomForestClassifier
from F2_MyAdaBoost import MyAdaBoostClassifier
from F2_MyGaussianProcess import Gaussian_Classifier
from F2_MyMLPClassifier import MLP_Classifier
from F2_My_KNN import KNN_Classifier

from F1_DecisionTreeClassifier import F1_MyDecisionTreeClassifier
from F1_MyRandomForest import F1_MyRandomForestClassifier
from F1_MyAdaBoost import F1_MyAdaBoostClassifier
from F1_MyGaussianProcess import F1_Gaussian_Classifier
from F1_MyMLPClassifier import F1_MLP_Classifier
from F1_My_KNN import F1_KNN_Classifier


from collections import OrderedDict
import os
import argparse
import random

def getDistancesFromCentre(centrePoint,fakeURLFeatureMatrix,distanceRange):
    from scipy.spatial import distance
    # Calculate Distances of each point from Centroid and store in a Dictionary...

    fakeURLDistances = OrderedDict()

    for i in range(0,len(fakeURLFeatureMatrix)):
        euclDistance = distance.euclidean(fakeURLFeatureMatrix[i],centrePoint)
        fakeURLDistances[i] = euclDistance

    #Sort Dictionary by Descending Order
    sorted_FakeURLDistances = sorted(fakeURLDistances.items(),key=lambda v:v[1],reverse=True)
    longIndexes = []

    for i in range(0,distanceRange):
        longIndexes.append(sorted_FakeURLDistances[i][0])

    return longIndexes




def parse_args():
    parser = argparse.ArgumentParser(description='PhishingURL-Oversampling')
    parser.add_argument('--dataset_file_name', type=str, default='None')
    parser.add_argument('--dataset_threshold', type=int, default=5)
    parser.add_argument('--fake_url_text', type=str, default='Unknown')
    parser.add_argument('--fake_url_threshold', type=int, default=5)
    parser.add_argument('--k', type=int, default=1500)
    parser.add_argument('--n_clusters', type=int, default=250)
    parser.add_argument('--kmeans_k', type=int, default=20)
    parser.add_argument('--random_samples',type = int, default=5000)
    parser.add_argument('--output_start', type=str, default='Unknown')
    parser.add_argument('--fake_features_file',type=str,default='None')
    return parser.parse_args()

# Create a CSV for threshold 10 for all 3 datasets
#args = parse_args()
# DATASET_FILE = args.dataset_file_name
# DATASET_THRESHOLD = 10
# FAKE_URL_TEXT = args.fake_url_text
# FAKE_URL_THRESHOLD = 10
# K = args.k
# N_CLUSTERS = args.n_clusters
# KMEANS_K = args.kmeans_k
# OUTPUT_START = args.output_start
# RANDOM_SAMPLES = args.random_samples
# #OUTRESPONSE_CSV_FILE = args.outresponseCSV
# FAKE_FEATURES_FILE = args.fake_features_file
dir_name = os.path.dirname(os.path.realpath(__file__))
#fileName = DATASET_FILE
datasetObject = LoadCSVDataset()
cleanedDataset,header = datasetObject.loadDataset("url_data.csv")
# #print cleanedDataset[header[0]][34001]
# #Here I've obtained the cleaned Dataset with N/A removed.
# #Below I've created methods to create a csv file with all the relevant String features.
featureObject = FeaturesCSV()
#threshold = DATASET_THRESHOLD
createdCSVFile = 'Features_url_data.csv'






#createdCSVFile = featureObject.createCSVFile(cleanedDataset,header,0,"url_data.csv")
print 'Loading Features of All URSs from CSV'
features = LoadFeatures()
featureMatrix,phishingLabel = features.loadFeatures(createdCSVFile)
#
# # phishyURLs = features.loadPositiveFeatures()
# # nonPhishyURLs = features.loadNegativeFeatures()
#
# phy = phishingLabel.count(1)
# nphy = phishingLabel.count(0)
#
# print 'Phishy:' + str(phy)
# print 'Non Phishy:' + str(nphy)
# pre = GanPreProcess()
#
# if len(phishyURLs) > len(nonPhishyURLs):
# 	for everyURL in nonPhishyURLs:				#Uncomment this if you need to Generate Training samples for GAN
# 		pre.preProcessURLs(everyURL)
# else:
# 	for everyURL in phishyURLs:
# 		pre.preProcessURLs(everyURL)

featureMatrix = numpy.array(featureMatrix,dtype='double')
phishingLabel = numpy.array(phishingLabel,dtype='double')







# ------------------------------ FOR GAN Techniques --------------------------------------------

# # ------------------------   Load the fake URL's and put them into a CSV file ----------------------
# features = LoadFeatures()
# # FakefileName = FAKE_URL_TEXT
# # fakeFeaturesFile = FAKE_FEATURES_FILE
# # # fakeCSV = LoadCSVDataset()
# # threshold2 = FAKE_URL_THRESHOLD
# # #fakeCSVFileName = 'EbayThreshold5NP_Cleaned_Outresponse.csv'
# # # fakeCSVFileName = fakeCSV.createCSVForFakeURL(FakefileName,threshold2)
# # # fakeURLcleanedDataset, fakeURLheader = fakeCSV.loadDataset(fakeCSVFileName)
# # # featurecsv = FeaturesCSV()
# # # featurecsv.createCSVFile(fakeURLcleanedDataset, fakeURLheader,threshold,fakeCSVFileName)
# # #--------------- Once you created a Fake CSV File, get it's feature matrix -------------------------
# #
# fakeURLFeatureMatrix,fakeURLphishingLabel = features.loadFeatures(fakeFeaturesFile)
# fakeURLFeatureMatrix = numpy.array(fakeURLFeatureMatrix,dtype='double')
# fakeURLphishingLabel = numpy.array(fakeURLphishingLabel, dtype = 'double')
# #
# #
# print len(fakeURLFeatureMatrix)
# print len(fakeURLphishingLabel)

#-------------- Apply Selection Techniques -----------------------------------


# F2 SCORES ---------------------------------------------------------------------------------------
#technique = 'Baseline'

# decisionTree7 =  MyDecisionTreeClassifier()
# decisionTree7.decisionTreeNoOversampling(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
# exit(1)
# decisionTree7 =  MyDecisionTreeClassifier()
# decisionTree7.decisionTreeNoOversampling(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# exit(1)
# decisionTree3 = MyDecisionTreeClassifier()
# decisionTree3.decisionTreebSMOTE2(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# decisionTree4 = MyDecisionTreeClassifier()
# decisionTree4.decisionTreeSVM_SMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# decisionTree5 = MyDecisionTreeClassifier()
# decisionTree5.decisionTreeADASYN(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# decisionTree6 = MyDecisionTreeClassifier()
# decisionTree6.decisionTreeRMR(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# decisionTree1 = MyDecisionTreeClassifier()
# decisionTree1.decisionTreeSMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# decisionTree2 = MyDecisionTreeClassifier()
# decisionTree2.decisionTreebSMOTE1(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
#
#
# rf1 = MyRandomForestClassifier()
# rf1.randomForestNoOversampling(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# rf2 = MyRandomForestClassifier()
# rf2.randomForestRMR(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# rf3 = MyRandomForestClassifier()
# rf3.randomForestSMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# rf4 = MyRandomForestClassifier()
# rf4.randomForestADASYN(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# rf5 = MyRandomForestClassifier()
# rf5.randomForestSVM_Smote(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# rf6 = MyRandomForestClassifier()
# rf6.randomForestb2SMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# rf7 = MyRandomForestClassifier()
# rf7.randomForestb1SMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
#
# ada1 = MyAdaBoostClassifier()
# ada1.adaBoostNoOversampling(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# ada2 = MyAdaBoostClassifier()
# ada2.adaBoostADASYN(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# ada3 = MyAdaBoostClassifier()
# ada3.adaBoostRMR(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# ada4 = MyAdaBoostClassifier()
# ada4.adaBoostSVMSmote(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# ada5 = MyAdaBoostClassifier()
# ada5.adaBoostb2SMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# ada6 = MyAdaBoostClassifier()
# ada6.adaBoostb1SMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# ada7 = MyAdaBoostClassifier()
# ada7.adaBoostSMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
#
#
# gau1 = Gaussian_Classifier()
# gau1.gaussianNoOversampling(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# gau2 = Gaussian_Classifier()
# gau2.gaussianADASYN(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# gau3 = Gaussian_Classifier()
# gau3.gaussianb2SMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# gau4 = Gaussian_Classifier()
# gau4.gaussianBoostb1SMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# gau5 = Gaussian_Classifier()
# gau5.gaussianBoostSMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# gau6 = Gaussian_Classifier()
# gau6.gaussianRMR(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# gau7 = Gaussian_Classifier()
# gau7.gaussianSVMSmote(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#

# mlp1 = MLP_Classifier()
# mlp1.mlpNoOversampling(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
# mlp2 = MLP_Classifier()
# mlp2.mlpADASYN(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# mlp3 = MLP_Classifier()
# mlp3.mlpb2SMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# mlp4 = MLP_Classifier()
# mlp4.mlpBoostb1SMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# mlp5 = MLP_Classifier()
# mlp5.mlpRMR(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# mlp6 = MLP_Classifier()
# mlp6.mlpSVMSmote(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# mlp7 = MLP_Classifier()
# mlp7.mlpBoostSMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#

# --------
# knn1 = KNN_Classifier()
# knn1.KNNADASYN(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# knn2 = KNN_Classifier()
# knn2.KNNb2SMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# knn3 = KNN_Classifier()
# knn3.KNNBoostb1SMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# knn4 = KNN_Classifier()
# knn4.KNNBoostSMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# knn5 = KNN_Classifier()
# knn5.KNNNoOversampling(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# knn6 = KNN_Classifier()
# knn6.KNNRMR(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# knn7 = KNN_Classifier()
# knn7.KNNSVMSmote(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# #F2 SCORES ENDS- --------------------------------------------------------------------------------------
#
#
# # NOW CALCULATE F1_SCORES
#
# # -------------- F1_SCORES --------------
# technique = 'Baseline'
# decisionTree7 =  F1_MyDecisionTreeClassifier()
# decisionTree7.decisionTreeNoOversampling(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# decisionTree3 = F1_MyDecisionTreeClassifier()
# decisionTree3.decisionTreebSMOTE2(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# decisionTree4 = F1_MyDecisionTreeClassifier()
# decisionTree4.decisionTreeSVM_SMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# decisionTree5 = F1_MyDecisionTreeClassifier()
# decisionTree5.decisionTreeADASYN(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# decisionTree6 = F1_MyDecisionTreeClassifier()
# decisionTree6.decisionTreeRMR(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# decisionTree1 = F1_MyDecisionTreeClassifier()
# decisionTree1.decisionTreeSMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# decisionTree2 = F1_MyDecisionTreeClassifier()
# decisionTree2.decisionTreebSMOTE1(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
#
#
# rf1 = F1_MyRandomForestClassifier()
# rf1.randomForestNoOversampling(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# rf2 = F1_MyRandomForestClassifier()
# rf2.randomForestRMR(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# rf3 = F1_MyRandomForestClassifier()
# rf3.randomForestSMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# rf4 = F1_MyRandomForestClassifier()
# rf4.randomForestADASYN(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# rf5 = F1_MyRandomForestClassifier()
# rf5.randomForestSVM_Smote(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# rf6 = F1_MyRandomForestClassifier()
# rf6.randomForestb2SMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# rf7 = F1_MyRandomForestClassifier()
# rf7.randomForestb1SMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
#
# ada1 = F1_MyAdaBoostClassifier()
# ada1.adaBoostNoOversampling(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# ada2 = F1_MyAdaBoostClassifier()
# ada2.adaBoostADASYN(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# ada3 = F1_MyAdaBoostClassifier()
# ada3.adaBoostRMR(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# ada4 = F1_MyAdaBoostClassifier()
# ada4.adaBoostSVMSmote(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# ada5 = F1_MyAdaBoostClassifier()
# ada5.adaBoostb2SMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# ada6 = F1_MyAdaBoostClassifier()
# ada6.adaBoostb1SMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# ada7 = F1_MyAdaBoostClassifier()
# ada7.adaBoostSMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
#
#
# gau1 = F1_Gaussian_Classifier()
# gau1.gaussianNoOversampling(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# gau2 = F1_Gaussian_Classifier()
# gau2.gaussianADASYN(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# gau3 = F1_Gaussian_Classifier()
# gau3.gaussianb2SMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# gau4 = F1_Gaussian_Classifier()
# gau4.gaussianBoostb1SMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# gau5 = F1_Gaussian_Classifier()
# gau5.gaussianBoostSMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# gau6 = F1_Gaussian_Classifier()
# gau6.gaussianRMR(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# gau7 = F1_Gaussian_Classifier()
# gau7.gaussianSVMSmote(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
#
# mlp1 = F1_MLP_Classifier()
# mlp1.mlpNoOversampling(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# mlp2 = F1_MLP_Classifier()
# mlp2.mlpADASYN(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# mlp3 = F1_MLP_Classifier()
# mlp3.mlpb2SMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# mlp4 = F1_MLP_Classifier()
# mlp4.mlpBoostb1SMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# mlp5 = F1_MLP_Classifier()
# mlp5.mlpRMR(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# mlp6 = F1_MLP_Classifier()
# mlp6.mlpSVMSmote(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# mlp7 = F1_MLP_Classifier()
# mlp7.mlpBoostSMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
# # --------
# knn1 = F1_KNN_Classifier()
# knn1.KNNADASYN(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# knn2 = F1_KNN_Classifier()
# knn2.KNNb2SMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# knn3 = F1_KNN_Classifier()
# knn3.KNNBoostb1SMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# knn4 = F1_KNN_Classifier()
# knn4.KNNBoostSMOTE(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# knn5 = F1_KNN_Classifier()
# knn5.KNNNoOversampling(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# knn6 = F1_KNN_Classifier()
# knn6.KNNRMR(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
# knn7 = F1_KNN_Classifier()
# knn7.KNNSVMSmote(featureMatrix,phishingLabel,0,0,technique,OUTPUT_START)
#
#
#---- Temp Exit



#-------------------------


# #--------------------------Technique 2. Choose k farthest Fake URLs ---------------------------------------------------
#
centroid = numpy.mean(fakeURLFeatureMatrix,axis=0)
from scipy.spatial import distance
# Calculate Distances of each point from Centroid and store in a Dictionary...

fakeURLDistances = OrderedDict()

for i in range(0,len(fakeURLFeatureMatrix)):
    euclDistance = distance.euclidean(fakeURLFeatureMatrix[i],centroid)
    fakeURLDistances[i] = euclDistance

#Sort Dictionary by Descending Order
sorted_FakeURLDistances = sorted(fakeURLDistances.items(),key=lambda v:v[1],reverse=True)
longIndexes = []

for i in range(0,K):
    longIndexes.append(sorted_FakeURLDistances[i][0])

finalFakeFeatures = []
for index in longIndexes:
    finalFakeFeatures.append(fakeURLFeatureMatrix[index])


fakeURLphishingLabel2 = fakeURLphishingLabel[:K]
technique = 'EucledianDistance'


# F-2 Scores Calculations---------------------------
# decisionTree7 =  MyDecisionTreeClassifier()
# decisionTree7.decisionTreeNoOversampling(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
# #
# decisionTree3 = MyDecisionTreeClassifier()
# decisionTree3.decisionTreebSMOTE2(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# decisionTree4 = MyDecisionTreeClassifier()
# decisionTree4.decisionTreeSVM_SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# decisionTree5 = MyDecisionTreeClassifier()
# decisionTree5.decisionTreeADASYN(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# decisionTree6 = MyDecisionTreeClassifier()
# decisionTree6.decisionTreeRMR(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# decisionTree1 = MyDecisionTreeClassifier()
# decisionTree1.decisionTreeSMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# decisionTree2 = MyDecisionTreeClassifier()
# decisionTree2.decisionTreebSMOTE1(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
#
#
# rf1 = MyRandomForestClassifier()
# rf1.randomForestNoOversampling(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# rf2 = MyRandomForestClassifier()
# rf2.randomForestRMR(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# rf3 = MyRandomForestClassifier()
# rf3.randomForestSMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# rf4 = MyRandomForestClassifier()
# rf4.randomForestADASYN(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# rf5 = MyRandomForestClassifier()
# rf5.randomForestSVM_Smote(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# rf6 = MyRandomForestClassifier()
# rf6.randomForestb2SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# rf7 = MyRandomForestClassifier()
# rf7.randomForestb1SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# ada1 = MyAdaBoostClassifier()
# ada1.adaBoostNoOversampling(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# ada2 = MyAdaBoostClassifier()
# ada2.adaBoostADASYN(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# ada3 = MyAdaBoostClassifier()
# ada3.adaBoostRMR(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# ada4 = MyAdaBoostClassifier()
# ada4.adaBoostSVMSmote(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# ada5 = MyAdaBoostClassifier()
# ada5.adaBoostb2SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# ada6 = MyAdaBoostClassifier()
# ada6.adaBoostb1SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# ada7 = MyAdaBoostClassifier()
# ada7.adaBoostSMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
#
# gau1 = Gaussian_Classifier()
# gau1.gaussianSVMSmote(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# gau2 = Gaussian_Classifier()
# gau2.gaussianRMR(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# gau3 = Gaussian_Classifier()
# gau3.gaussianBoostSMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# gau4 = Gaussian_Classifier()
# gau4.gaussianBoostb1SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# gau5 = Gaussian_Classifier()
# gau5.gaussianb2SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# gau6 = Gaussian_Classifier()
# gau6.gaussianADASYN(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# gau7 = Gaussian_Classifier()
# gau7.gaussianNoOversampling(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#

mlp1 = MLP_Classifier()
mlp1.mlpBoostSMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)

mlp2 = MLP_Classifier()
mlp2.mlpSVMSmote(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)

mlp3 = MLP_Classifier()
mlp3.mlpRMR(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)

mlp4 = MLP_Classifier()
mlp4.mlpBoostb1SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)

mlp5 = MLP_Classifier()
mlp5.mlpb2SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)

mlp6 = MLP_Classifier()
mlp6.mlpADASYN(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)

mlp7 = MLP_Classifier()
mlp7.mlpNoOversampling(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)

# knn1 = KNN_Classifier()
# knn1.KNNSVMSmote(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# knn2 = KNN_Classifier()
# knn2.KNNBoostSMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# knn3 = KNN_Classifier()
# knn3.KNNRMR(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# knn4 = KNN_Classifier()
# knn4.KNNNoOversampling(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# knn5 = KNN_Classifier()
# knn5.KNNBoostb1SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# knn6 = KNN_Classifier()
# knn6.KNNb2SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# knn7 = KNN_Classifier()
# knn7.KNNADASYN(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#

# -------- Eucledian Distance F-2 ENDS --------

#---------- Calculate F1
# decisionTree7 =  F1_MyDecisionTreeClassifier()
# decisionTree7.decisionTreeNoOversampling(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
# #
# decisionTree3 = F1_MyDecisionTreeClassifier()
# decisionTree3.decisionTreebSMOTE2(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# decisionTree4 = F1_MyDecisionTreeClassifier()
# decisionTree4.decisionTreeSVM_SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# decisionTree5 = F1_MyDecisionTreeClassifier()
# decisionTree5.decisionTreeADASYN(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# decisionTree6 = F1_MyDecisionTreeClassifier()
# decisionTree6.decisionTreeRMR(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# decisionTree1 = F1_MyDecisionTreeClassifier()
# decisionTree1.decisionTreeSMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# decisionTree2 = F1_MyDecisionTreeClassifier()
# decisionTree2.decisionTreebSMOTE1(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
#
#
# rf1 = F1_MyRandomForestClassifier()
# rf1.randomForestNoOversampling(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# rf2 = F1_MyRandomForestClassifier()
# rf2.randomForestRMR(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# rf3 = F1_MyRandomForestClassifier()
# rf3.randomForestSMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# rf4 = F1_MyRandomForestClassifier()
# rf4.randomForestADASYN(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# rf5 = F1_MyRandomForestClassifier()
# rf5.randomForestSVM_Smote(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# rf6 = F1_MyRandomForestClassifier()
# rf6.randomForestb2SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# rf7 = F1_MyRandomForestClassifier()
# rf7.randomForestb1SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# ada1 = F1_MyAdaBoostClassifier()
# ada1.adaBoostNoOversampling(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# ada2 = F1_MyAdaBoostClassifier()
# ada2.adaBoostADASYN(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# ada3 = F1_MyAdaBoostClassifier()
# ada3.adaBoostRMR(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# ada4 = F1_MyAdaBoostClassifier()
# ada4.adaBoostSVMSmote(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# ada5 = F1_MyAdaBoostClassifier()
# ada5.adaBoostb2SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# ada6 = F1_MyAdaBoostClassifier()
# ada6.adaBoostb1SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# ada7 = F1_MyAdaBoostClassifier()
# ada7.adaBoostSMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
#
# gau1 = F1_Gaussian_Classifier()
# gau1.gaussianSVMSmote(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# gau2 = F1_Gaussian_Classifier()
# gau2.gaussianRMR(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# gau3 = F1_Gaussian_Classifier()
# gau3.gaussianBoostSMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# gau4 = F1_Gaussian_Classifier()
# gau4.gaussianBoostb1SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# gau5 = F1_Gaussian_Classifier()
# gau5.gaussianb2SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# gau6 = F1_Gaussian_Classifier()
# gau6.gaussianADASYN(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# gau7 = F1_Gaussian_Classifier()
# gau7.gaussianNoOversampling(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#

mlp1 = F1_MLP_Classifier()
mlp1.mlpBoostSMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)

mlp2 = F1_MLP_Classifier()
mlp2.mlpSVMSmote(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)

mlp3 = F1_MLP_Classifier()
mlp3.mlpRMR(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)

mlp4 = F1_MLP_Classifier()
mlp4.mlpBoostb1SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)

mlp5 = F1_MLP_Classifier()
mlp5.mlpb2SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)

mlp6 = F1_MLP_Classifier()
mlp6.mlpADASYN(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)

mlp7 = F1_MLP_Classifier()
mlp7.mlpNoOversampling(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)

# knn1 = F1_KNN_Classifier()
# knn1.KNNSVMSmote(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# knn2 = F1_KNN_Classifier()
# knn2.KNNBoostSMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# knn3 = F1_KNN_Classifier()
# knn3.KNNRMR(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# knn4 = F1_KNN_Classifier()
# knn4.KNNNoOversampling(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# knn5 = F1_KNN_Classifier()
# knn5.KNNBoostb1SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# knn6 = F1_KNN_Classifier()
# knn6.KNNb2SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)
#
# knn7 = F1_KNN_Classifier()
# knn7.KNNADASYN(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel2,technique,OUTPUT_START)

# F1  ends (Eucledian)



# #*****************************************************************************************************************
#
# # -------------------------- Technique 3: Selection using K-means clustering ------------------------------------
technique = 'kMeans'
# Apply K-means clustering algorithm on the fakeURLFeatureMatrix. Select k = 250. Then for each
# obatined centroid, calculate Distances and take 18 farthest points
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters=N_CLUSTERS,random_state=10).fit(fakeURLFeatureMatrix)
kMeansFakeMatrix = []
for everyCenter in kmeans.cluster_centers_:
    #Calculate the distances from each point in feature matrix and sort it
    indexes = getDistancesFromCentre(everyCenter,fakeURLFeatureMatrix,KMEANS_K)
    for everyIndex in indexes:
        kMeansFakeMatrix.append(fakeURLFeatureMatrix[everyIndex])

kMeansFakeLabels = fakeURLphishingLabel[0:len(kMeansFakeMatrix)]

print 'PRINTINT LENG'
print len(kMeansFakeMatrix)
print len(kMeansFakeLabels)

print len(featureMatrix)
print len(phishingLabel)


# -------------------- KMEANS --------------------
# decisionTree7 =  MyDecisionTreeClassifier()
# decisionTree7.decisionTreeNoOversampling(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
# #
# decisionTree3 = MyDecisionTreeClassifier()
# decisionTree3.decisionTreebSMOTE2(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# decisionTree4 = MyDecisionTreeClassifier()
# decisionTree4.decisionTreeSVM_SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# decisionTree5 = MyDecisionTreeClassifier()
# decisionTree5.decisionTreeADASYN(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# decisionTree6 = MyDecisionTreeClassifier()
# decisionTree6.decisionTreeRMR(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# decisionTree1 = MyDecisionTreeClassifier()
# decisionTree1.decisionTreeSMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# decisionTree2 = MyDecisionTreeClassifier()
# decisionTree2.decisionTreebSMOTE1(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
#
#
# rf1 = MyRandomForestClassifier()
# rf1.randomForestNoOversampling(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# rf2 = MyRandomForestClassifier()
# rf2.randomForestRMR(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# rf3 = MyRandomForestClassifier()
# rf3.randomForestSMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# rf4 = MyRandomForestClassifier()
# rf4.randomForestADASYN(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# rf5 = MyRandomForestClassifier()
# rf5.randomForestSVM_Smote(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# rf6 = MyRandomForestClassifier()
# rf6.randomForestb2SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# rf7 = MyRandomForestClassifier()
# rf7.randomForestb1SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# ada1 = MyAdaBoostClassifier()
# ada1.adaBoostNoOversampling(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# ada2 = MyAdaBoostClassifier()
# ada2.adaBoostADASYN(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# ada3 = MyAdaBoostClassifier()
# ada3.adaBoostRMR(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# ada4 = MyAdaBoostClassifier()
# ada4.adaBoostSVMSmote(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# ada5 = MyAdaBoostClassifier()
# ada5.adaBoostb2SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# ada6 = MyAdaBoostClassifier()
# ada6.adaBoostb1SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# ada7 = MyAdaBoostClassifier()
# ada7.adaBoostSMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#

# gau1 = Gaussian_Classifier()
# gau1.gaussianSVMSmote(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# gau2 = Gaussian_Classifier()
# gau2.gaussianRMR(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# gau3 = Gaussian_Classifier()
# gau3.gaussianBoostSMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# gau4 = Gaussian_Classifier()
# gau4.gaussianBoostb1SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# gau5 = Gaussian_Classifier()
# gau5.gaussianb2SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# gau6 = Gaussian_Classifier()
# gau6.gaussianADASYN(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# gau7 = Gaussian_Classifier()
# gau7.gaussianNoOversampling(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)


mlp1 = MLP_Classifier()
mlp1.mlpBoostSMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
                                         technique,OUTPUT_START)

mlp2 = MLP_Classifier()
mlp2.mlpSVMSmote(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
                                         technique,OUTPUT_START)

mlp3 = MLP_Classifier()
mlp3.mlpRMR(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
                                         technique,OUTPUT_START)

mlp4 = MLP_Classifier()
mlp4.mlpBoostb1SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
                                         technique,OUTPUT_START)

mlp5 = MLP_Classifier()
mlp5.mlpb2SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
                                         technique,OUTPUT_START)

mlp6 = MLP_Classifier()
mlp6.mlpADASYN(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
                                         technique,OUTPUT_START)

mlp7 = MLP_Classifier()
mlp7.mlpNoOversampling(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
                                         technique,OUTPUT_START)

# knn1 = KNN_Classifier()
# knn1.KNNSVMSmote(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# knn2 = KNN_Classifier()
# knn2.KNNBoostSMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# knn3 = KNN_Classifier()
# knn3.KNNRMR(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# knn4 = KNN_Classifier()
# knn4.KNNNoOversampling(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# knn5 = KNN_Classifier()
# knn5.KNNBoostb1SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# knn6 = KNN_Classifier()
# knn6.KNNb2SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# knn7 = KNN_Classifier()
# knn7.KNNADASYN(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#

# -------- Eucledian Distance F-2 ENDS --------

#---------- Calculate F1
# decisionTree7 =  F1_MyDecisionTreeClassifier()
# decisionTree7.decisionTreeNoOversampling(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
# #
# decisionTree3 = F1_MyDecisionTreeClassifier()
# decisionTree3.decisionTreebSMOTE2(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# decisionTree4 = F1_MyDecisionTreeClassifier()
# decisionTree4.decisionTreeSVM_SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# decisionTree5 = F1_MyDecisionTreeClassifier()
# decisionTree5.decisionTreeADASYN(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# decisionTree6 = F1_MyDecisionTreeClassifier()
# decisionTree6.decisionTreeRMR(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# decisionTree1 = F1_MyDecisionTreeClassifier()
# decisionTree1.decisionTreeSMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# decisionTree2 = F1_MyDecisionTreeClassifier()
# decisionTree2.decisionTreebSMOTE1(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
#
#
# rf1 = F1_MyRandomForestClassifier()
# rf1.randomForestNoOversampling(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# rf2 = F1_MyRandomForestClassifier()
# rf2.randomForestRMR(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# rf3 = F1_MyRandomForestClassifier()
# rf3.randomForestSMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# rf4 = F1_MyRandomForestClassifier()
# rf4.randomForestADASYN(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# rf5 = F1_MyRandomForestClassifier()
# rf5.randomForestSVM_Smote(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# rf6 = F1_MyRandomForestClassifier()
# rf6.randomForestb2SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# rf7 = F1_MyRandomForestClassifier()
# rf7.randomForestb1SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# ada1 = F1_MyAdaBoostClassifier()
# ada1.adaBoostNoOversampling(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# ada2 = F1_MyAdaBoostClassifier()
# ada2.adaBoostADASYN(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# ada3 = F1_MyAdaBoostClassifier()
# ada3.adaBoostRMR(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# ada4 = F1_MyAdaBoostClassifier()
# ada4.adaBoostSVMSmote(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# ada5 = F1_MyAdaBoostClassifier()
# ada5.adaBoostb2SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# ada6 = F1_MyAdaBoostClassifier()
# ada6.adaBoostb1SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# ada7 = F1_MyAdaBoostClassifier()
# ada7.adaBoostSMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
#
# gau1 = F1_Gaussian_Classifier()
# gau1.gaussianSVMSmote(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# gau2 = F1_Gaussian_Classifier()
# gau2.gaussianRMR(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# gau3 = F1_Gaussian_Classifier()
# gau3.gaussianBoostSMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# gau4 = F1_Gaussian_Classifier()
# gau4.gaussianBoostb1SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# gau5 = F1_Gaussian_Classifier()
# gau5.gaussianb2SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# gau6 = F1_Gaussian_Classifier()
# gau6.gaussianADASYN(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# gau7 = F1_Gaussian_Classifier()
# gau7.gaussianNoOversampling(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#

mlp1 = F1_MLP_Classifier()
mlp1.mlpBoostSMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
                                         technique,OUTPUT_START)

mlp2 = F1_MLP_Classifier()
mlp2.mlpSVMSmote(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
                                         technique,OUTPUT_START)

mlp3 = F1_MLP_Classifier()
mlp3.mlpRMR(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
                                         technique,OUTPUT_START)

mlp4 = F1_MLP_Classifier()
mlp4.mlpBoostb1SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
                                         technique,OUTPUT_START)

mlp5 = F1_MLP_Classifier()
mlp5.mlpb2SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
                                         technique,OUTPUT_START)

mlp6 = F1_MLP_Classifier()
mlp6.mlpADASYN(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
                                         technique,OUTPUT_START)

mlp7 = F1_MLP_Classifier()
mlp7.mlpNoOversampling(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
                                         technique,OUTPUT_START)

# knn1 = F1_KNN_Classifier()
# knn1.KNNSVMSmote(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# knn2 = F1_KNN_Classifier()
# knn2.KNNBoostSMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# knn3 = F1_KNN_Classifier()
# knn3.KNNRMR(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# knn4 = F1_KNN_Classifier()
# knn4.KNNNoOversampling(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# knn5 = F1_KNN_Classifier()
# knn5.KNNBoostb1SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# knn6 = F1_KNN_Classifier()
# knn6.KNNb2SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)
#
# knn7 = F1_KNN_Classifier()
# knn7.KNNADASYN(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
#                                          technique,OUTPUT_START)

# F1  ends (Kmeans)


#___________________KMEANS ENDS __________________________





# # ***************************************************************************************************************

randomFakeFeatures = random.sample(fakeURLFeatureMatrix,RANDOM_SAMPLES)
randomFakeLabels = fakeURLphishingLabel[:len(randomFakeFeatures)]

technique = 'RandomSampling'
decisionTree7 =  MyDecisionTreeClassifier()
decisionTree7.decisionTreeNoOversampling(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# decisionTree3 = MyDecisionTreeClassifier()
# decisionTree3.decisionTreebSMOTE2(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# decisionTree4 = MyDecisionTreeClassifier()
# decisionTree4.decisionTreeSVM_SMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# decisionTree5 = MyDecisionTreeClassifier()
# decisionTree5.decisionTreeADASYN(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# decisionTree6 = MyDecisionTreeClassifier()
# decisionTree6.decisionTreeRMR(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# decisionTree1 = MyDecisionTreeClassifier()
# decisionTree1.decisionTreeSMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# decisionTree2 = MyDecisionTreeClassifier()
# decisionTree2.decisionTreebSMOTE1(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
#
#
# rf1 = MyRandomForestClassifier()
# rf1.randomForestNoOversampling(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# rf2 = MyRandomForestClassifier()
# rf2.randomForestRMR(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# rf3 = MyRandomForestClassifier()
# rf3.randomForestSMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# rf4 = MyRandomForestClassifier()
# rf4.randomForestADASYN(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# rf5 = MyRandomForestClassifier()
# rf5.randomForestSVM_Smote(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# rf6 = MyRandomForestClassifier()
# rf6.randomForestb2SMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# rf7 = MyRandomForestClassifier()
# rf7.randomForestb1SMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# ada1 = MyAdaBoostClassifier()
# ada1.adaBoostNoOversampling(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# ada2 = MyAdaBoostClassifier()
# ada2.adaBoostADASYN(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# ada3 = MyAdaBoostClassifier()
# ada3.adaBoostRMR(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# ada4 = MyAdaBoostClassifier()
# ada4.adaBoostSVMSmote(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# ada5 = MyAdaBoostClassifier()
# ada5.adaBoostb2SMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# ada6 = MyAdaBoostClassifier()
# ada6.adaBoostb1SMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# ada7 = MyAdaBoostClassifier()
# ada7.adaBoostSMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# gau1 = Gaussian_Classifier()
# gau1.gaussianSVMSmote(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# gau2 = Gaussian_Classifier()
# gau2.gaussianRMR(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# gau3 = Gaussian_Classifier()
# gau3.gaussianBoostSMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# gau4 = Gaussian_Classifier()
# gau4.gaussianBoostb1SMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# gau5 = Gaussian_Classifier()
# gau5.gaussianb2SMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# gau6 = Gaussian_Classifier()
# gau6.gaussianADASYN(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# gau7 = Gaussian_Classifier()
# gau7.gaussianNoOversampling(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#

mlp1 = MLP_Classifier()
mlp1.mlpBoostSMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)

mlp2 = MLP_Classifier()
mlp2.mlpSVMSmote(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)

mlp3 = MLP_Classifier()
mlp3.mlpRMR(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)

mlp4 = MLP_Classifier()
mlp4.mlpBoostb1SMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)

mlp5 = MLP_Classifier()
mlp5.mlpb2SMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)

mlp6 = MLP_Classifier()
mlp6.mlpADASYN(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)

mlp7 = MLP_Classifier()
mlp7.mlpNoOversampling(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)

# knn1 = KNN_Classifier()
# knn1.KNNSVMSmote(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# knn2 = KNN_Classifier()
# knn2.KNNBoostSMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# knn3 = KNN_Classifier()
# knn3.KNNRMR(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# knn4 = KNN_Classifier()
# knn4.KNNNoOversampling(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# knn5 = KNN_Classifier()
# knn5.KNNBoostb1SMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# knn6 = KNN_Classifier()
# knn6.KNNb2SMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# knn7 = KNN_Classifier()
# knn7.KNNADASYN(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
#
# # F1 scores for Random Fake
# decisionTree7 =  F1_MyDecisionTreeClassifier()
# decisionTree7.decisionTreeNoOversampling(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
# #
# decisionTree3 = F1_MyDecisionTreeClassifier()
# decisionTree3.decisionTreebSMOTE2(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# decisionTree4 = F1_MyDecisionTreeClassifier()
# decisionTree4.decisionTreeSVM_SMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# decisionTree5 = F1_MyDecisionTreeClassifier()
# decisionTree5.decisionTreeADASYN(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# decisionTree6 = F1_MyDecisionTreeClassifier()
# decisionTree6.decisionTreeRMR(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# decisionTree1 = F1_MyDecisionTreeClassifier()
# decisionTree1.decisionTreeSMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# decisionTree2 = F1_MyDecisionTreeClassifier()
# decisionTree2.decisionTreebSMOTE1(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
#
#
# rf1 = F1_MyRandomForestClassifier()
# rf1.randomForestNoOversampling(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# rf2 = F1_MyRandomForestClassifier()
# rf2.randomForestRMR(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# rf3 = F1_MyRandomForestClassifier()
# rf3.randomForestSMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# rf4 = F1_MyRandomForestClassifier()
# rf4.randomForestADASYN(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# rf5 = F1_MyRandomForestClassifier()
# rf5.randomForestSVM_Smote(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# rf6 = F1_MyRandomForestClassifier()
# rf6.randomForestb2SMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# rf7 = F1_MyRandomForestClassifier()
# rf7.randomForestb1SMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# ada1 = F1_MyAdaBoostClassifier()
# ada1.adaBoostNoOversampling(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# ada2 = F1_MyAdaBoostClassifier()
# ada2.adaBoostADASYN(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# ada3 = F1_MyAdaBoostClassifier()
# ada3.adaBoostRMR(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# ada4 = F1_MyAdaBoostClassifier()
# ada4.adaBoostSVMSmote(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# ada5 = F1_MyAdaBoostClassifier()
# ada5.adaBoostb2SMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# ada6 = F1_MyAdaBoostClassifier()
# ada6.adaBoostb1SMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# ada7 = F1_MyAdaBoostClassifier()
# ada7.adaBoostSMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# gau1 = F1_Gaussian_Classifier()
# gau1.gaussianSVMSmote(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# gau2 = F1_Gaussian_Classifier()
# gau2.gaussianRMR(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# gau3 = F1_Gaussian_Classifier()
# gau3.gaussianBoostSMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# gau4 = F1_Gaussian_Classifier()
# gau4.gaussianBoostb1SMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# gau5 = F1_Gaussian_Classifier()
# gau5.gaussianb2SMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# gau6 = F1_Gaussian_Classifier()
# gau6.gaussianADASYN(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# gau7 = F1_Gaussian_Classifier()
# gau7.gaussianNoOversampling(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#

mlp1 = F1_MLP_Classifier()
mlp1.mlpBoostSMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)

mlp2 = F1_MLP_Classifier()
mlp2.mlpSVMSmote(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)

mlp3 = F1_MLP_Classifier()
mlp3.mlpRMR(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)

mlp4 = F1_MLP_Classifier()
mlp4.mlpBoostb1SMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)

mlp5 = F1_MLP_Classifier()
mlp5.mlpb2SMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)

mlp6 = F1_MLP_Classifier()
mlp6.mlpADASYN(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)

mlp7 = F1_MLP_Classifier()
mlp7.mlpNoOversampling(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)

# knn1 = F1_KNN_Classifier()
# knn1.KNNSVMSmote(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# knn2 = F1_KNN_Classifier()
# knn2.KNNBoostSMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# knn3 = F1_KNN_Classifier()
# knn3.KNNRMR(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# knn4 = F1_KNN_Classifier()
# knn4.KNNNoOversampling(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# knn5 = F1_KNN_Classifier()
# knn5.KNNBoostb1SMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# knn6 = F1_KNN_Classifier()
# knn6.KNNb2SMOTE(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)
#
# knn7 = F1_KNN_Classifier()
# knn7.KNNADASYN(featureMatrix,phishingLabel,randomFakeFeatures,randomFakeLabels,technique,OUTPUT_START)

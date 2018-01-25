from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
class Resampling():

    # ------------- SMOTE Oversampling -------------------------------------

    def smoteOversampling(self,featureMatrix, Labels):
        sm = SMOTE(kind='regular')
        #print type(featureMatrix[0][0])
        #print type(Labels[0])
        feature_Resampled, Labels_Resampled = sm.fit_sample(featureMatrix, Labels)
        #print type(feature_Resampled[0][0])
        #print type(Labels_Resampled[0])
        print "SMOTE Oversampling completed....."
        return feature_Resampled,Labels_Resampled

    #---------------- Borderline SMOTE 1 Oversampling -----------------------

    def b1smoteOversampling(self,featureMatrix, Labels):
        sm = SMOTE(kind='borderline1')
        feature_Resampled, Labels_Resampled = sm.fit_sample(featureMatrix, Labels)
        print "Borderline-1 SMOTE Oversampling completed"
        return feature_Resampled,Labels_Resampled


    #--------------- Borderline SMOTE 2 Oversampling -------------------------

    def b2smoteOversampling(self,featureMatrix, Labels):
        sm = SMOTE(kind='borderline2')
        #print type(featureMatrix[0][0])
        #print type(Labels[0])
        feature_Resampled, Labels_Resampled = sm.fit_sample(featureMatrix, Labels)
        #print type(feature_Resampled[0][0])
        #print type(Labels_Resampled[0])
        print "Borderline-2 SMOTE Oversampling completed...."
        return feature_Resampled,Labels_Resampled


    #-------- Support Vector Machine SMOTE Oversampling -----------------------------

    def SVMsmoteOversampling(self,featureMatrix, Labels):
        sm = SMOTE(kind='svm')
        #print type(featureMatrix[0][0])
        #print type(Labels[0])
        feature_Resampled, Labels_Resampled = sm.fit_sample(featureMatrix, Labels)
        #print type(feature_Resampled[0][0])
        #print type(Labels_Resampled[0])
        print "SVM SMOTE Oversampling completed..."
        return feature_Resampled,Labels_Resampled

    # -------------- Adaptive Synthetic Sampling approach for imbalanced learning ----------------
    def ADASYNOversampling(self,featureMatrix, Labels):
        ada = ADASYN(random_state=42)
        #print type(featureMatrix[0][0])
        #print type(Labels[0])
        feature_Resampled, Labels_Resampled = ada.fit_sample(featureMatrix, Labels)
        #print type(feature_Resampled[0][0])
        #print type(Labels_Resampled[0])
        print "ADASYN Oversampling Completed"
        return feature_Resampled,Labels_Resampled


    #----------------- Randome Minority oversampling with replacement -------------------


    def RMROversampling(self,featureMatrix, Labels):
        ror = RandomOverSampler(random_state=42)
        #print type(featureMatrix[0][0])
        #print type(Labels[0])
        feature_Resampled, Labels_Resampled = ror.fit_sample(featureMatrix, Labels)
        #print type(feature_Resampled[0][0])
        #print type(Labels_Resampled[0])
        print "Random Minority with Replacement Oversampling completed..."
        return feature_Resampled,Labels_Resampled
import os
from sklearn.model_selection import train_test_split
import pandas as pd
class Hunt():
    def hunt(self):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/'+ 'HuntingResultsAll.txt', 'a+')
        predictionResult.truncate()
        # ----- 1. Handle Ebay Dataset ----

        realFiles = ['Features_T5_Outresponse-ebay.csv','Features_T8_Outresponse-ebay.csv','Features_T10_Outresponse-ebay.csv',
                    'Features_T5_Outresponse-Bofa.csv','Features_T8_Outresponse-Bofa.csv','Features_T10_Outresponse-Bofa.csv',
                    'Features_T5_Outresponse-Paypal.csv','Features_T8_Outresponse-Paypal.csv','Features_T10_Outresponse-Paypal.csv']

        fakeFiles = ['Features_T5_Ebay_T5_NP_Cleaned_Outresponse.csv','Features_T8_Ebay_T8_P_Cleaned_Outresponse.csv','Features_T10_Ebay_T10_P_Cleaned_Outresponse.csv',
                     'Features_T5_Bofa_T5_NP_Cleaned_Outresponse.csv','Features_T8_Bofa_T8_P_Cleaned_Outresponse.csv','Features_T10_Bofa_T10_P_Cleaned_Outresponse.csv',
                     'Features_T5_Paypal_T5_NP_Cleaned_Outresponse.csv','Features_T8_Paypal_T8_P_Cleaned_Outresponse.csv','Features_T10_Paypal_T10_P_Cleaned_Outresponse.csv']


        for i in range(0,len(realFiles)):
            real = pd.read_csv(dir_name+'/'+realFiles[i])
            fake = pd.read_csv(dir_name+'/'+fakeFiles[i])

            real = real['URL'].sample(frac=0.22)
            real = real['URL'].tolist()
            fake = fake['URL'].tolist()
            huntedURLs = []
            count = 0
            huntedURLs,count = self.hunting(real,fake)
            predictionResult.write('Total hunted in '+realFiles[i]+'--->'+str(count))
            predictionResult.write('List of Hunted URLs in ' + realFiles[i]+ '-->'+str(huntedURLs))


    def hunting(self,real,fake):
        counter = 0
        hunted = []
        for everyURL in fake:
            if everyURL in real:
                counter+=1
                hunted.append(everyURL)

        return hunted,counter

#Create a CSV File denoted relevant String features
import socket
import re
import csv
import os
import scipy.stats
import re
from collections import defaultdict
import csv
class FeaturesCSV():
    'This class generates a csv File with relevant String features'


    def validateIPAddress(self,URL):
        isValidIp = '0'
        try:
            hostStart = URL.index("//")
            hostEnd = URL.index("/", hostStart + 2)
        except:
            hostEnd = len(URL)
        host = URL[hostStart+2:hostEnd]
        #Pattern match the Host part for IP address or Hexadecimal places
        try:
            socket.inet_aton(host)  #handles hexadecimal encoded IP address as well....
            #legal..return flag should be True
            isValidIp = '1'
        except:
            isValidIp = '0'

        return isValidIp

    def isLongURL(self,URL):
        'Consult for Ternary Values'
        isLongURL = '0'
        if len(URL) < 54:
            isLongURL = '0'
        elif len(URL) >= 54 and len(URL) <=75:
            isLongURL = '1'
        else:
            isLongURL = '2'
        return isLongURL

    def dash_count_in_path(self,URL):
        dash_count = 0

        try:
            hostStart = URL.index("//")
            hostEnd = URL.index("/", hostStart + 2)
        except:
            hostEnd = len(URL)

        path = URL[hostEnd+1:]
        dash_count = path.count('-')

        return dash_count

    def vowel_consonant_ratio(self,URL):
        vowels = ['a','e','i','o','u','A','E','I','O','U']
        consonants = ['b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','w','x','y','z'
                      ,'B','C','D','F','G','H','J','K','L','M','N','P','Q','R','S','T','V','W','X','Y','Z']

        vowConsonant_ratio = 0
        number_of_consonants = sum(URL.count(c) for c in consonants)

        number_of_vowels = sum(URL.count(c) for c in vowels)
        try:
            vowConsonant_ratio = number_of_vowels // number_of_consonants
        except Exception as e:
            vowConsonant_ratio = 0  # if divide by 0 error occurs

        return vowConsonant_ratio

    def digit_letter_ratio_URL(self,URL):
        digitLetterRatio = 0
        digits = ['0','1','2','3','4','5','6','7','8','9']
        num_of_digits = sum(URL.count(x) for x in digits)
        no_of_letters = len(re.findall('[a-zA-Z]',URL))
        try:
            digitLetterRatio = no_of_letters // num_of_digits
        except Exception as e:
            #Divide by 0 error may occur
            digitLetterRatio = 0

        return digitLetterRatio


    def brand_name_present_dash(self,URL):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        columns = defaultdict(list)
        csvFile = open(dir_name + '/top500_domains.csv', 'a+')
        reader = csv.reader(csvFile)
        header = next(reader)

        with open (dir_name+'/top500_domains.csv') as file:
            reader = csv.DictReader(file)
            for row in reader:
                for(k,v) in row.items():
                    columns[k].append(v)
        brand_domains = columns[header[1]]
        present = 0
        for x in brand_domains:
            if x+'-' in URL:
                present = 1
                break
            if '-'+x in URL:
                present = 1
                break

        return present


    def very_short_hostname(self,URL):
        try:
            hostStart = URL.index("//")
            hostEnd = URL.index("/", hostStart + 2)
        except:
            hostEnd = len(URL)
        host = URL[hostStart + 2:hostEnd]
        if len(host) <=5:
            return 1
        else:
            return 0

    def colons_in_hostname(self,URL):
        ''' Indicative of port no. manipulation'''
        try:
            hostStart = URL.index("//")
            hostEnd = URL.index("/", hostStart + 2)
        except:
            hostEnd = len(URL)
        host = URL[hostStart + 2:hostEnd]
        count = host.count(':')
        return count



    def preSuffixInURL(self,URL):
        isPreSuffix = '0'
        try:
            hostStart = URL.index("//")
            hostEnd = URL.index("/", hostStart + 2)
        except:
            hostEnd = len(URL)
        host = URL[hostStart + 2:hostEnd]
        count = str(host).count('-')
        if count > 0:
            isPreSuffix = '1'
        return isPreSuffix

    def subDomain(self,URL):
        'check if Ternary'
        isMultipleDomains = '0'
        try:
            hostStart = URL.index("//")
            hostEnd = URL.index("/", hostStart + 2)
        except:
            hostEnd = len(URL)
        host = URL[hostStart + 2:hostEnd]
        #including www., if the dots are greater than 3, then High!
        count = str(host).count('.')
        if count < 3:
            isMultipleDomains = '0'
        elif count == 3:
            isMultipleDomains = '1'
        else:
            isMultipleDomains = '2'

        return isMultipleDomains

    def checkSymbol(self,URL):
        isSymbol = '0'
        #check if @ in host part
        try:
            hostStart = URL.index("//")
            hostEnd = URL.index("/", hostStart + 2)
        except:
            hostEnd = len(URL)

        host = URL[hostStart + 2:hostEnd]
        if str(host).find("@") > 0:
            isSymbol = '1'
        return isSymbol

    #Check for HTTPS feature. If included without checking issuer, there will be high false positives

    def topLevelDomainCount(self,URL):
        'counts the occurences of top level domains by matching regular expressions'
        topLevelDomain = '0'
        try:
            hostStart = URL.index("//")
            hostEnd = URL.index("/", hostStart + 2)
        except:
            hostEnd = len(URL)

        path = URL[hostEnd+1:]
        m = re.compile(r'\.([^.\n\s]*)$', re.M)
        f = re.findall(m, path)
        if len(f) > 0:
            topLevelDomain = '1'

        return topLevelDomain

    def suspicousWords(self,URL):
        'Counts certain suspicious words....'
        haveSuspicious = '0'
        suspicousDatabase = ["confirm","account","secure","ebayisapi","webscr","login","signin","submit","update","logon","wp","cmd","admin"]
        count=0
        for everySuspiciousKeyword in suspicousDatabase:
            if everySuspiciousKeyword in URL:
                count+=1
        if count>1:
            haveSuspicious = '1'
        return haveSuspicious


    def countPunctuation(self,URL):
        'Counts certain punctuation marks'
        punctuationFeature = '0'
        blacklistedPunctuations = ['!','#','$','*',';',':','\'']
        count = 0
        for everPunctuation in blacklistedPunctuations:
            if everPunctuation in URL:
                count+=1
        if count > 1:
            punctuationFeature = '1'

        return punctuationFeature

    def digitsInDomain(self,URL):
        isDigits = '0'
        try:
            hostStart = URL.index("//")
            hostEnd = URL.index("/", hostStart + 2)
        except:
            hostEnd = len(URL)
        try:
            host = URL[hostStart + 2:hostEnd]
            numbers = re.search(r'\d+', host).group()
        except:
            #no numbers found
            numbers = 0
            isDigits = '0'

        if numbers > 0:
            isDigits = '1'
        return isDigits


    def long_host_name(self,URL):
        try:
            hostStart = URL.index("//")
            hostEnd = URL.index("/", hostStart + 2)
        except:
            hostEnd = len(URL)
        host = URL[hostStart + 2:hostEnd]
        if len(host) > 22:
            return 1
        else:
            return 0


    def getCharacterFrequency(self,URL):
        import collections
        freq = collections.Counter(URL)
        freqSorted = sorted(freq.items())
        freqList = []
        for i in range(0, 26):
            freqList.append(0)
        for key, value in freqSorted:
            if key.isalpha():
                #check for
                freqList[ord(key.lower()) - 97] = int(value)
        return freqList

    def getEntropy(self,URL):
        freqList = self.getCharacterFrequency(URL)
        entropy = scipy.stats.entropy(freqList)
        return entropy

    def getKLDivergence(self,URL):
        freqEnglish = [8.12, 1.49, 2.71, 4.32, 12.02, 2.30, 2.03, 5.92, 7.31, 0.10, 0.69, 3.98, 2.61, 6.95, 7.68, 1.82, 0.11,
             6.02, 6.28, 9.10, 2.98, 1.11, 2.09, 0.17, 2.11, 0.07]
        freqList = self.getCharacterFrequency(URL)
        kld = scipy.stats.entropy(freqList,freqEnglish)
        return kld


    def createCSVFile(self,columns,originalHeader,threshold,fileName):
        'Creates a CSV File denoting features of the URL'
        dir_name = os.path.dirname(os.path.realpath(__file__))
        createdFile = 'Features_'+fileName
        with open(dir_name+'/'+createdFile, 'wb') as featureCSVFile:
            w = csv.writer(featureCSVFile)
            w.writerow(["URL","IP", "LongURL", "PreSuffix","SubDomain","@Symbol","TLDInPath","SuspiciousWords",
                        "PunctuationSymbols","DigitsInDomain","Entropy","KLDivergence",
                        "DashCountPath","V/C","D/L","BrandName-","ShortHostname","LongHostName",
                        "PortManipulation","Phishy", "Time"])
            count = 0
            for everyURL in columns[originalHeader[0]]:
                features = []
                features.append(everyURL)
                features.append(self.validateIPAddress(everyURL))
                features.append(self.isLongURL(everyURL))
                features.append(self.preSuffixInURL(everyURL))
                features.append(self.subDomain(everyURL))
                features.append(self.checkSymbol(everyURL))
                features.append(self.topLevelDomainCount(everyURL))
                features.append(self.suspicousWords(everyURL))
                features.append(self.countPunctuation(everyURL))
                features.append(self.digitsInDomain(everyURL))
                features.append(self.getEntropy(everyURL))
                features.append(self.getKLDivergence(everyURL))
                features.append(self.dash_count_in_path(everyURL))
                features.append(self.vowel_consonant_ratio(everyURL))
                features.append(self.digit_letter_ratio_URL(everyURL))
                features.append(self.brand_name_present_dash(everyURL))
                features.append(self.very_short_hostname(everyURL))
                features.append(self.long_host_name(everyURL))
                features.append(self.colons_in_hostname(everyURL))
                features.append(columns[originalHeader[1]][count])
                features.append(columns[originalHeader[2]][count])
                count += 1
                #print columns[originalHeader[0]][count]
                # if int(columns[originalHeader[1]][count]) >= threshold:
                #     #then phishy
                #     count+=1
                #     #print 'Phishy ratio'
                #     features.append("1")
                # else:
                #     features.append("0")
                #     count+=1
                #print features
                #write these features to the csv File
                w.writerow(features)
            return createdFile

    def normalized(self,lst):
        s = sum(lst)
        return map(lambda x: float(x) / s, lst)



# a_normalized =obj.normalized(a)
# print 'done'
# b_normalized = obj.normalized(b)
# print a_normalized
# print b_normalized
# print "%.2f" % scipy.stats.entropy(a_normalized) #low entropy signifies meaningless string. Normalizes internally
# print scipy.stats.entropy(a,b) # Calculates K-L divergence. Normalizes internally
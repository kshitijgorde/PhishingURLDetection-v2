class GanPreProcess():

    def preProcessURLs(self, URL):
        #URL = "http://www.idahotherapy.com/XM/X/DHL/pu1671ng971b92pjx4pdqajk.php?rand=13InboxLightaspxn.1774256418&fid&1252899642&fid.1&fav.1"
        # i = 0
        # i = URL.find("//")
        # newURL = URL.split("/")
        # print newURL
        with open("GanPreprocess.txt","a+") as f:

            #f.write(newURL[0]+'//')
            #newURL = newURL[1:]
            try:
                hostStart = URL.index("//")
                hostEnd = URL.index("/", hostStart + 2)
            except:
                hostEnd = len(URL)

            protocol = URL[0:hostStart]
            path = URL[hostEnd + 1:]
            host = URL[hostStart+2:hostEnd]
            #print "Separating Protocol, Host and Path"
            f.write(protocol+'//')
            self.processHost(host,f)

            pathParts = path.split('/')
            for part in pathParts:
                self.processPath(part,f)

            f.write('\n')
            # for i in range(len(newURL)):
            #     part = newURL[i]
            #
            #     if str(part) != '':
            #         #Split by dots first...check if dots present
            #         dotFlag = str(part).find(".")
            #         if dotFlag == -1:
            #
            #             continue
            #             #No dots found.....Process with others
            #         else:
            #             self.processDots(str(part),f,hostFlag)

    def processHost(self, host,fileHandle):
        writeBuffer = []
        #1. Host may contain a Port number
        isPort = host.find(":")
        if isPort != -1:
            portNo = host[isPort:]
            host = host[0:isPort]
            writeBuffer.append(' '+str(portNo))


        dotparts = host.split(".")
        # Process in reverse
        if 'html' in host or 'php' in host or 'htm' in host or 'jsp' in host or 'asp' in host:
            writeBuffer.append(' ' + str(host))
            dotparts = []

        for i in reversed(dotparts):
            #print i
            if i == dotparts[-1]:
                writeBuffer.append(' ' + i)
            else:
                writeBuffer.append(' ' + i + '.')

        for string in reversed(writeBuffer):
            fileHandle.write(string)



    def processPath(self, dotPart,fileHandle):
        writeBuffer = []
        if dotPart!= '':
            dotPart = '/'+dotPart
        dotparts = dotPart.split(".")

        #Process in reverse
        if 'html' in dotPart or 'php' in dotPart or 'htm' in dotPart or 'jsp' in dotPart or 'asp' in dotPart:
            isParameter = dotPart.find("?")
            if isParameter != -1:
                # found
                fileName = dotPart[0:isParameter]
                fileHandle.write(' '+str(fileName))
                #dotPart = dotPart[isParameter:]
                isEqual = dotPart.find("=")
                if isEqual != -1:
                    value = dotPart[isParameter:isEqual + 1]
                    fileHandle.write(' ' + value)
                    dotPart = dotPart[isEqual + 1:]
                    #Check for & and split
                    isAmp = dotPart.split("&")

                    if isAmp != -1:
                        for everyField in isAmp:
                            fileHandle.write(' '+'&'+everyField)
                            dotPart =  ' '


            # check @ in i..
                isAt = dotPart.find("@")
                if isAt != -1:
                    atValue = dotPart[0:isAt + 1]
                    fileHandle.write(' ' + atValue)
                    i = dotPart[isAt + 1:]

            writeBuffer.append(' ' + str(dotPart))
            dotparts = []

        for i in reversed(dotparts):
            # check for ? (indicates parameters)\
            isParameter = i.find("?")
            if isParameter!=-1:
                #found
                isEqual = i.find("=")
                if isEqual != -1:
                    value = i[isParameter:isEqual+1]
                    fileHandle.write(' '+value)
                    i = i[isEqual+1:]

            #check @ in i..
            isAt = i.find("@")
            if isAt != -1:
                atValue = i[0:isAt+1]
                fileHandle.write(' '+atValue)
                i = i[isAt+1:]

            if i == dotparts[-1]:
                writeBuffer.append(' '+i)
            else:
                writeBuffer.append(' '+i+'.')

        for string in reversed(writeBuffer):
            fileHandle.write(string)

import sys
import random
import math


'''Get feature data from file as a matrix with a row per data instance'''
def getFeatureData(featureFile):
    x=[]
    dFile = open(featureFile, 'r')
    for line in dFile:
        rVec=[] 
        row = line.split()
        rVec.append(float(1))
        for item in row:
            rVec.append(float(item))
        x.append(rVec)
    dFile.close()
    return x

'''Get label data from file as a dictionary with key as data instance indexand value as the class index'''
def getLabelData(labelFile):   
    lFile = open(labelFile, 'r')  
    lDict = {}  
    for line in lFile: 
        row = line.split() 
        lDict[row[1]] = int(row[0]) 
    lFile.close()  
    return lDict



dataFileName = sys.argv[1]
data = getFeatureData(dataFileName)
labelFileName = sys.argv[2]
lDict = getLabelData(labelFileName)
eta=float(sys.argv[3])
stopc=float(sys.argv[4])



rows=(len(data))
cols=(len(data[0]))
#random number generator to initilaize values for w
w = []
for j in range(cols):
    w.append(random.random() * 0.02 - 0.01)
    
print("w=",w)


#Logistic discrimination
def get_ld():
    temp_1=0
    sum_1 = []
    for i in range(0,rows):
        sum_1.append(0) 
    for i in range(0,rows):
        for j in range(0,cols):
            sum_1[i] = sum_1[i] + (w[j] * data[i][j])   
    for i in range(0,len(lDict)):
        if (lDict.get(str(i))!= None):
            temp_1 = temp_1 + (-lDict[str(i)]*math.log(1/(1 + math.exp(-sum_1[i])))) - (1-lDict[str(i)]) * math.log((math.exp(-sum_1[i]))/(1+math.exp(-sum_1[i])))
    return temp_1



#compution of weight
def get_weight():
    dw = []
    dw1=[]
    for i in range(0, rows):
        dw.append(0)
        dw1.append(0)
    for i in range(0, rows):
        for j in range(0, cols):
            dw[i] = dw[i] + data[i][j]*w[j]
    for i in range(0,len(lDict)):
        if lDict.get(str(i))!=None:
            dw[i] = 1/(1+math.exp(-1*dw[i]))
    for i in range(0, len(lDict)):
        for j in range(0, cols):
            if lDict.get(str(i))!= None:
                dw1[j] = dw1[j] + ((lDict[str(i)] - dw[i]) * data[i][j])

    for i in range(cols):
        w[i] = w[i] + (eta * dw1[i])
  
count =0
prevError = get_ld()
while(True):
    count = count +1
    get_weight()
    ld = get_ld()
    if(abs(prevError - ld) <= stopc):
       break
    prevError = ld

#distance to origin
d = 0
w0 = w[0]
l2_norm= 0
for i in range(1,cols):
    l2_norm = l2_norm + w[i]**2
l2_norm= l2_norm**0.5
print("||w||= ",l2_norm)
print("w = ")
for i in range(1,cols):
    d = d + w[i] **2
    print(w[i])
distance = w0/(math.sqrt(d))
print ("Distance to origin =",distance)

#predicting training labels
for i in range(0,rows):
    p = 0
    if (lDict.get(str(i)) == None):  
        for j in range(0,cols):
            p = p + (w[j] * data[i][j])
        if(p < 0):
            print("0", i)
        else:
            print("1", i)

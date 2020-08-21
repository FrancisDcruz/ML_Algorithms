import sys
import random
import math

#Retrival of feature data from file as matrix
def getFeatureData(featureFile):
    x=[]
    dFile = open(featureFile, 'r')
    for line in dFile:
        rVec=[] 
        row = line.split()
        for item in row:
            rVec.append(float(item))
        rVec.append(float(1))
        x.append(rVec)
    dFile.close()
    return x

#Retrival label data from file as a dictionary with key
def getLabelData(labelFile): 
    lFile = open(labelFile, 'r')  
    lDict = {} 
    for line in lFile: 
        row = line.split() 
        lDict[row[1]] = int(row[0]) 
        if(lDict[row[1]]==0):
            lDict[row[1]]=int(-1)
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

w = []
for j in range(cols):
    w.append(0.02 * random.random() - 0.01)
    


#calculation of dot product
def dot_product(a, b):
    dtp = 0
    for i in range(cols):
        dtp += a[i] * b[i]
    return dtp



#Gradient decent
hinge_l = rows * 10
diff = 1
count = 0
while ((diff) > stopc):
    d = [0]*cols    
    for j in range(rows):
        if (lDict.get(str(j)) != None):
            dtp = dot_product(w, data[j])
            condition = (lDict.get(str(j)) * (dot_product(w, data[j])))
            for k in range(cols):
                if (condition < 1):
                    d[k] += -1 * ((lDict.get(str(j))) * data[j][k])
                else:
                    d[k] += 0


#Updating of weight w
    for j in range(cols):
        w[j] = w[j] - eta * d[j]
    prev_error= hinge_l
    hinge_l= 0


#computing hingloss(hinge_l)
    for j in range(rows):
        if (lDict.get(str(j)) != None):
            hinge_l += max(0, 1 - (lDict.get(str(j)) * dot_product(w, data[j])))
        diff= abs(prev_error-hinge_l)
        
    print ("hinge loss = " + str(hinge_l))

normw = 0
for i in range(cols - 1):
    normw += w[i] ** 2
    
normw = math.sqrt(normw)

dist_orgin = abs(w[len(w) - 1] / normw)
print ("Distance to  the origin = " + str(dist_orgin))

#Prediction of values
for i in range(rows):
    if (lDict.get(str(i)) == None):
        dtp = dot_product(w, data[i])
        if (dtp > 0):
            print("1 " + str(i))
        else:
            print("0 " + str(i))    